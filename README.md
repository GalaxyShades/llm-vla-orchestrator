# LLM-VLA Orchestrator

Minimal scaffold for a multimodal-LLM + VLA closed loop with sequential subtasks.

## Pipeline mode

The orchestrator is configured for `chess_move`: adaptive chess orchestration with legal transition checks and Stockfish-driven move policy.

## Install

Quick setup (recommended):

```bash
./setup.sh
```

Manual setup:

```bash
conda env create -f envs/environment.yml
conda activate llm-vla-orchestrator
```

To sync an existing environment to the spec:

```bash
conda env update -n llm-vla-orchestrator -f envs/environment.yml --prune
```

Install a new package ad hoc (without editing `requirements.txt` yet):

```bash
python -m pip install <package-name>
```

The recommended interpreter for this repo is:
- `/home/lem/miniconda3/envs/llm-vla-orchestrator/bin/python`

## Run chess orchestration API

Start the backend service:

```bash
/home/lem/miniconda3/envs/llm-vla-orchestrator/bin/python -m orchestrator.run \
  --config configs/chess_move.yaml \
  --serve-api
```

This launches FastAPI on `http://127.0.0.1:8000` with:
- `POST /api/player/analyse`: analyse and process a player's completed move
- `POST /api/reset`: reset game state
- `GET /api/state`: current game state, pending-warning state, and per-game UI cache
- `POST /api/ui/state`: persist frontend UI cache (events, last result, timers) for the active game
- `WS /ws/events`: live event stream

The backend uses camera-based analysis and ChatGPT vision to infer SAN from `before_fen` + after-move image.

Camera input source is configurable in `configs/chess_move.yaml`:
- `chess.camera.input_mode: filesystem` uses `chess.camera.inbox_dir/current_filename`
- `chess.camera.input_mode: ui_render` uses the rendered Chess Camera snapshot sent by the frontend UI

Policy-agent config resolution:
- `chess.orchestrator_agent.model` falls back to `AZURE_AGENT_DEPLOYMENT`
- `chess.orchestrator_agent.api_key` falls back to `AZURE_AGENT_API_KEY`
- `chess.orchestrator_agent.base_url` falls back to `AZURE_AGENT_ENDPOINT` (converted to `/openai/v1`)
- `chess.orchestrator_agent.api_version` falls back to `AZURE_AGENT_API_VERSION`
- `chess.orchestrator_agent.azure_endpoint` falls back to `AZURE_AGENT_ENDPOINT`

Vision config resolution:
- `chess.vision.model` falls back to `AZURE_VISION_DEPLOYMENT`
- `chess.vision.api_key` falls back to `AZURE_VISION_API_KEY`
- `chess.vision.base_url` falls back to `AZURE_VISION_ENDPOINT` (converted to `/openai/v1`)
- `chess.vision.api_version` falls back to `AZURE_VISION_API_VERSION`
- `chess.vision.azure_endpoint` falls back to `AZURE_VISION_ENDPOINT`
- `chess.vision.azure_model_allowlist` defines Azure-routed deployment names (comma-separated)
- `chess.vision.hf_max_retries` configures Hugging Face retry attempts for rate-limit resilience
- `chess.vision.hf_timeout_s` configures Hugging Face request timeout in seconds
- `HUGGING_FACE_API_KEY` is used for non-`gpt-*` vision models via Hugging Face chat completions

For one-shot CLI debugging (without UI):

```bash
/home/lem/miniconda3/envs/llm-vla-orchestrator/bin/python -m orchestrator.run \
  --config configs/chess_move.yaml \
  --observed-piece-placement "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR"
```

To reset state before a run:

```bash
/home/lem/miniconda3/envs/llm-vla-orchestrator/bin/python -m orchestrator.run \
  --config configs/chess_move.yaml \
  --reset-game-state
```

## Hardware execution

The chess pipeline calls `executor.execute_move(<uci>)` after selecting the AI
move. Execution is configured in `configs/chess_move.yaml` under
`chess.execution`.

Default dry-run mode:

```yaml
chess:
  execution:
    type: dry_run
```

This logs text such as `Move piece from e2 to e4` and does not move hardware.

To enable Cobot Magic execution, first calibrate board-square joint targets in
`configs/cobot_chess_square_poses.yaml`, then switch:

```yaml
chess:
  execution:
    type: cobot
    cobot_module_path: cobot_magic/mobile_aloha_AgileX
    square_poses_path: configs/cobot_chess_square_poses.yaml
    active_arm: left
    move_method: continuous
```

The square-pose file expects 7D joint targets:

```yaml
squares:
  e2: [joint0, joint1, joint2, joint3, joint4, joint5, gripper]
  e4: [joint0, joint1, joint2, joint3, joint4, joint5, gripper]
```

With `type: cobot`, a selected move like `e2e4` is translated into the
calibrated `e2` pose followed by the calibrated `e4` pose and published through
Cobot Magic's ROS bridge.

For ROS, CAN, and Piper launch instructions, see
[cobot_magic/README.md](cobot_magic/README.md).

The chess pipeline now:
- validates observed board transitions with `python-chess`
- assumes legal player intent and infers the most likely legal transition when observation is inconsistent
- estimates player strength using a rolling move window
- selects AI responses with Stockfish under adaptive difficulty policy
- logs per-move policy context, engine candidates, and execution records

### Backend module layout

- `orchestrator/game_service.py`: end-to-end move orchestration flow
- `orchestrator/vision_agent.py`: ChatGPT vision SAN move extraction
- `orchestrator/engine_service.py`: Stockfish analysis and candidate generation
- `orchestrator/policy_agent.py`: `ChessOrchestratorAgent` LangChain tool-calling move policy
- `orchestrator/difficulty.py`: player Elo estimation and policy targets
- `orchestrator/executor.py`: dry-run and Cobot Magic execution adapters
- `orchestrator/game_state.py`: persistent game state store
- `orchestrator/game_logger.py`: per-move artefacts and JSONL/PGN logging
- `orchestrator/camera.py`: directory-based camera image source
- `orchestrator/chess_types.py`: shared dataclasses used across modules

### Frontend UI (React)

The frontend lives in `frontend/` and uses `react-chessboard` + `chess.js`.

```bash
cd frontend
npm install
npm run dev
```

UI controls:
- `Start Game` / `Reset Game`
- Drag-and-drop legal player moves (move completion auto-triggers analysis)
- `Player View` / `Chess Camera` toggle controls which board rendering is captured and sent for analysis logging
- `Vision model` dropdown to choose the active model for analysis (`gpt-5.3-chat` default)

### Important caveat about FEN

The camera path uses the canonical pre-move FEN from game memory plus the after-move image,
then asks vision for SAN move notation.
This avoids reconstructing full FEN fields directly from an image and keeps legal-state handling
anchored to `python-chess` move validation.

`api_version` is optional in this code.
- If `api_version` is set (agent or vision), the code uses `AzureChatOpenAI` with `azure_endpoint` and `api_version`.
- If `api_version` is blank, the code uses `ChatOpenAI` in Azure OpenAI v1-style `base_url` mode.
Use separate env vars for each client:
- `AZURE_AGENT_API_KEY`, `AZURE_AGENT_ENDPOINT`, `AZURE_AGENT_API_VERSION`
- `AZURE_VISION_API_KEY`, `AZURE_VISION_ENDPOINT`, `AZURE_VISION_API_VERSION`

## LangSmith tracing

Tracing can be configured via YAML + `.env`.

- Config toggle: `langsmith.enabled` (true/false)
- Project name: `LANGSMITH_PROJECT` in `.env`
- Credentials: `LANGSMITH_API_KEY` in `.env`

Optional `.env` fields:

```bash
LANGSMITH_API_KEY=...
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
```

When enabled, traces include:
- chess orchestration decisions
- vision analysis calls
- Stockfish evaluation and move selection
- Azure OpenAI API calls (via wrapped client)

## Output artifacts

Each chess game writes:
- `games/<game_date_time>/moves.jsonl` (one file per game)
- `games/<game_date_time>/game.pgn` (one PGN per game)
- `games/<game_date_time>/moves/move_XXX/move_XXX_pre.png`
- `games/<game_date_time>/moves/move_XXX/move_XXX_observed.png`
- `games/<game_date_time>/moves/move_XXX/move_XXX_post.png`
- `games/<game_date_time>/moves/move_XXX/move_XXX_<source>.png` (camera snapshot used for analysis)

Each completed player move writes to its own directory: `moves/move_001`, `moves/move_002`, and so on.

## Vision benchmark harness

Generate random chess transitions and benchmark multiple vision models:

```bash
# one-time browser install for snapshot capture
cd frontend && npx playwright install chromium && cd ..

python -m orchestrator.benchmark_vision_models \
  --config configs/chess_move.yaml \
  --samples 20 \
  --models "gpt-5.3-chat,gpt-4o,Qwen/Qwen2.5-VL-7B-Instruct,zai-org/GLM-4.5V"
```

Outputs in `data/benchmark_vision/`:
- `samples.jsonl`: generated ground-truth benchmark set
- `images/sample_XXX.png`: rendered from the frontend `ChessBoard3D` view (same default pitch/zoom)
- `vision_predicitions.csv`: per-sample per-model predictions and legality checks
- `vision_summary.csv`: aggregate metrics (`accuracy`, `success rate`, `runtime` excluding failed calls, retries, confidence)
