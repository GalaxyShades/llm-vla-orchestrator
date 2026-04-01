import { useEffect, useMemo, useRef, useState } from "react";
import { Chess } from "chess.js";
import { Chessboard } from "react-chessboard";
import ChessBoard3D from "./ChessBoard3D";

const API_BASE = import.meta.env.VITE_API_BASE || "";
const UI_CACHE_KEY_PREFIX = "chess_ui_cache:";
const FILES = ["a", "b", "c", "d", "e", "f", "g", "h"];
const RANKS = [8, 7, 6, 5, 4, 3, 2, 1];

function pieceCodeToFenChar(pieceCode) {
  const map = {
    wP: "P",
    wN: "N",
    wB: "B",
    wR: "R",
    wQ: "Q",
    wK: "K",
    bP: "p",
    bN: "n",
    bB: "b",
    bR: "r",
    bQ: "q",
    bK: "k"
  };
  return map[pieceCode] || "";
}

function fenCharToPieceCode(char) {
  const map = {
    P: "wP",
    N: "wN",
    B: "wB",
    R: "wR",
    Q: "wQ",
    K: "wK",
    p: "bP",
    n: "bN",
    b: "bB",
    r: "bR",
    q: "bQ",
    k: "bK"
  };
  return map[char] || null;
}

function fenToPiecePlacement(fen) {
  return String(fen || "").trim().split(" ")[0];
}

function piecePlacementToBoardPosition(piecePlacement) {
  const position = {};
  const rows = piecePlacement.split("/");
  for (let rankIdx = 0; rankIdx < rows.length; rankIdx += 1) {
    const rank = RANKS[rankIdx];
    let fileIdx = 0;
    for (const token of rows[rankIdx]) {
      if (/\d/.test(token)) {
        fileIdx += Number(token);
      } else {
        const square = `${FILES[fileIdx]}${rank}`;
        const pieceCode = fenCharToPieceCode(token);
        if (pieceCode) {
          position[square] = pieceCode;
        }
        fileIdx += 1;
      }
    }
  }
  return position;
}

function boardPositionToPiecePlacement(position) {
  const ranks = [];
  for (const rank of RANKS) {
    let row = "";
    let emptyCount = 0;
    for (const file of FILES) {
      const square = `${file}${rank}`;
      const pieceCode = position[square];
      if (!pieceCode) {
        emptyCount += 1;
      } else {
        if (emptyCount > 0) {
          row += String(emptyCount);
          emptyCount = 0;
        }
        row += pieceCodeToFenChar(pieceCode);
      }
    }
    if (emptyCount > 0) {
      row += String(emptyCount);
    }
    ranks.push(row || "8");
  }
  return ranks.join("/");
}

function getErrorMessage(error) {
  if (!error) {
    return "";
  }
  if (typeof error === "string") {
    return error;
  }
  if (error.message) {
    return error.message;
  }
  return "Unknown error";
}

function getStatusTone(statusText) {
  if (statusText === "Player move") {
    return "ok";
  }
  if (statusText === "AI move") {
    return "info";
  }
  if (statusText === "Game over") {
    return "done";
  }
  return "idle";
}

function waitMs(durationMs) {
  return new Promise((resolve) => {
    window.setTimeout(resolve, durationMs);
  });
}

function getApiUrl(path) {
  if (API_BASE) {
    return `${API_BASE}${path}`;
  }
  return path;
}

function getWebSocketUrl(path) {
  if (API_BASE) {
    const apiUrl = new URL(API_BASE);
    const wsProtocol = apiUrl.protocol === "https:" ? "wss:" : "ws:";
    return `${wsProtocol}//${apiUrl.host}${path}`;
  }
  const wsProtocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${wsProtocol}//${window.location.host}${path}`;
}

function uiCacheKey(gameId) {
  return `${UI_CACHE_KEY_PREFIX}${gameId}`;
}

function readLocalUiState(gameId) {
  if (!gameId) {
    return null;
  }
  try {
    const raw = window.localStorage.getItem(uiCacheKey(gameId));
    if (!raw) {
      return null;
    }
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object") {
      return null;
    }
    return parsed;
  } catch {
    return null;
  }
}

function writeLocalUiState(gameId, payload) {
  if (!gameId) {
    return;
  }
  window.localStorage.setItem(uiCacheKey(gameId), JSON.stringify(payload));
}

function updatedAtMs(value) {
  const timeMs = Date.parse(String(value || ""));
  if (!Number.isFinite(timeMs)) {
    return 0;
  }
  return timeMs;
}

export default function App() {
  const boardFrameRef = useRef(null);
  const uiRenderFrameRef = useRef(null);
  const uiHydratedRef = useRef(false);
  const [boardPosition, setBoardPosition] = useState(() =>
    piecePlacementToBoardPosition("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR")
  );
  const [playerMoveStartedAt, setPlayerMoveStartedAt] = useState(null);
  const [aiMoveStartedAt, setAiMoveStartedAt] = useState(null);
  const [playerLastMoveSeconds, setPlayerLastMoveSeconds] = useState(0);
  const [aiLastMoveSeconds, setAiLastMoveSeconds] = useState(0);
  const [playerTotalSeconds, setPlayerTotalSeconds] = useState(0);
  const [aiTotalSeconds, setAiTotalSeconds] = useState(0);
  const [statusText, setStatusText] = useState("Awaiting game start");
  const [hasStarted, setHasStarted] = useState(false);
  const [clockNowMs, setClockNowMs] = useState(Date.now());
  const [currentFen, setCurrentFen] = useState("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
  const [lastResult, setLastResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [warning, setWarning] = useState("");
  const [error, setError] = useState("");
  const [aiReason, setAiReason] = useState("");
  const [eventFeed, setEventFeed] = useState([]);
  const [pendingAnalysis, setPendingAnalysis] = useState(null);
  const [visionBypassPayload, setVisionBypassPayload] = useState(null);
  const [gameId, setGameId] = useState("");
  const [viewMode, setViewMode] = useState("2d");
  const [cameraInputMode, setCameraInputMode] = useState("filesystem");
  const [cameraPitchDeg, setCameraPitchDeg] = useState(65);
  const [cameraDistance, setCameraDistance] = useState(18);

  const CAMERA_MIN_PITCH_DEG = 8;
  const CAMERA_MAX_PITCH_DEG = 88;

  const piecePlacement = useMemo(
    () => boardPositionToPiecePlacement(boardPosition),
    [boardPosition]
  );
  const statusTone = useMemo(() => getStatusTone(statusText), [statusText]);
  const canInteract = useMemo(
    () => (statusText === "Player move" || statusText === "Awaiting game start") && !isLoading,
    [isLoading, statusText]
  );

  const formatTimer = (secondsFloat) => {
    const elapsedSeconds = Math.max(0, Math.floor(secondsFloat));
    const minutes = Math.floor(elapsedSeconds / 60);
    const seconds = elapsedSeconds % 60;
    const mm = String(minutes).padStart(2, "0");
    const ss = String(seconds).padStart(2, "0");
    return `${mm}:${ss}`;
  };

  const playerCurrentSeconds = useMemo(() => {
    if (!hasStarted || playerMoveStartedAt === null || statusText !== "Player move") {
      return 0;
    }
    return Math.max(0, (clockNowMs - playerMoveStartedAt) / 1000);
  }, [clockNowMs, hasStarted, playerMoveStartedAt, statusText]);

  const aiCurrentSeconds = useMemo(() => {
    if (!hasStarted || aiMoveStartedAt === null || statusText !== "AI move") {
      return 0;
    }
    return Math.max(0, (clockNowMs - aiMoveStartedAt) / 1000);
  }, [aiMoveStartedAt, clockNowMs, hasStarted, statusText]);

  const playerTimerText = useMemo(
    () => formatTimer(playerCurrentSeconds),
    [playerCurrentSeconds]
  );
  const aiTimerText = useMemo(() => formatTimer(aiCurrentSeconds), [aiCurrentSeconds]);
  const playerLastMoveText = useMemo(
    () => formatTimer(playerLastMoveSeconds),
    [playerLastMoveSeconds]
  );
  const aiLastMoveText = useMemo(() => formatTimer(aiLastMoveSeconds), [aiLastMoveSeconds]);
  const playerTotalText = useMemo(() => formatTimer(playerTotalSeconds), [playerTotalSeconds]);
  const aiTotalText = useMemo(() => formatTimer(aiTotalSeconds), [aiTotalSeconds]);
  const playerElapsedSeconds = useMemo(() => {
    if (!hasStarted || playerMoveStartedAt === null) {
      return 0;
    }
    return Math.max(0, (clockNowMs - playerMoveStartedAt) / 1000);
  }, [clockNowMs, hasStarted, playerMoveStartedAt]);

  useEffect(() => {
    if (
      !hasStarted ||
      (statusText !== "Player move" && statusText !== "AI move") ||
      (statusText === "Player move" && playerMoveStartedAt === null) ||
      (statusText === "AI move" && aiMoveStartedAt === null)
    ) {
      return undefined;
    }
    const timer = setInterval(() => {
      setClockNowMs(Date.now());
    }, 100);
    return () => {
      clearInterval(timer);
    };
  }, [aiMoveStartedAt, hasStarted, playerMoveStartedAt, statusText]);

  useEffect(() => {
    let isMounted = true;

    async function bootstrapState() {
      try {
        const response = await fetch(getApiUrl("/api/state"));
        if (!response.ok) {
          throw new Error("Could not fetch initial state");
        }
        const data = await response.json();
        if (!isMounted) {
          return;
        }
        const loadedGameId = String(data.game_id || "");
        setGameId(loadedGameId);
        const currentFen = String(data.current_fen || "");
        const initialFen = String(data.initial_fen || currentFen);
        let loadedCameraInputMode = String(data.camera_input_mode || "filesystem");
        if (loadedCameraInputMode === "frontend_ui") {
          loadedCameraInputMode = "ui_render";
        }
        const loadedPlacement = fenToPiecePlacement(currentFen);
        const initialPlacement = fenToPiecePlacement(initialFen);
        setBoardPosition(piecePlacementToBoardPosition(loadedPlacement));
        setCurrentFen(currentFen);
        setCameraInputMode(loadedCameraInputMode);

        const loadedBoard = new Chess(currentFen);
        const hasGameProgress =
          Number(data.move_index || 0) > 0 || loadedPlacement !== initialPlacement;
        if (loadedBoard.isGameOver()) {
          setHasStarted(true);
          setStatusText("Game over");
          setPlayerMoveStartedAt(null);
          setAiMoveStartedAt(null);
          setClockNowMs(Date.now());
        } else if (hasGameProgress) {
          setHasStarted(true);
          setStatusText("Player move");
          // Unknown wall-clock on reload; restart from 00:00.
          setPlayerMoveStartedAt(Date.now());
          setAiMoveStartedAt(null);
          setClockNowMs(Date.now());
        } else {
          setHasStarted(false);
          setStatusText("Awaiting game start");
          setPlayerMoveStartedAt(null);
          setAiMoveStartedAt(null);
          setClockNowMs(Date.now());
        }
        setPlayerLastMoveSeconds(0);
        setAiLastMoveSeconds(0);
        setPlayerTotalSeconds(0);
        setAiTotalSeconds(0);
        setWarning("");

        const backendUiState =
          data.ui_state && typeof data.ui_state === "object" ? data.ui_state : null;
        const localUiState = readLocalUiState(loadedGameId);
        const candidates = [backendUiState, localUiState].filter(
          (entry) => entry && String(entry.game_id || "") === loadedGameId
        );
        const syncedUiState = candidates.sort(
          (left, right) => updatedAtMs(right.updated_at) - updatedAtMs(left.updated_at)
        )[0];
        if (syncedUiState) {
          setHasStarted(Boolean(syncedUiState.has_started));
          setStatusText(String(syncedUiState.status_text || "Awaiting game start"));
          setLastResult(syncedUiState.last_result || null);
          setAiReason(
            String(
              syncedUiState.ai_reason ||
                syncedUiState.last_result?.ai_reason ||
                ""
            )
          );
          setEventFeed(
            Array.isArray(syncedUiState.event_feed) ? syncedUiState.event_feed.slice(0, 20) : []
          );
          setPlayerLastMoveSeconds(Math.max(0, Number(syncedUiState.player_last_move_seconds || 0)));
          setAiLastMoveSeconds(Math.max(0, Number(syncedUiState.ai_last_move_seconds || 0)));
          setPlayerTotalSeconds(Math.max(0, Number(syncedUiState.player_total_seconds || 0)));
          setAiTotalSeconds(Math.max(0, Number(syncedUiState.ai_total_seconds || 0)));
          if (String(syncedUiState.status_text || "") === "Player move") {
            setPlayerMoveStartedAt(Date.now());
            setAiMoveStartedAt(null);
          } else if (String(syncedUiState.status_text || "") === "AI move") {
            setAiMoveStartedAt(Date.now());
            setPlayerMoveStartedAt(null);
          }
        }
        uiHydratedRef.current = true;
      } catch (err) {
        if (isMounted) {
          setError(getErrorMessage(err));
        }
      }
    }

    bootstrapState();

    return () => {
      isMounted = false;
    };
  }, []);

  useEffect(() => {
    if (!gameId || !uiHydratedRef.current) {
      return;
    }
    const payload = {
      game_id: gameId,
      updated_at: new Date().toISOString(),
      status_text: statusText,
      has_started: hasStarted,
      ai_reason: aiReason,
      last_result: lastResult,
      event_feed: eventFeed.slice(0, 20),
      player_last_move_seconds: playerLastMoveSeconds,
      ai_last_move_seconds: aiLastMoveSeconds,
      player_total_seconds: playerTotalSeconds,
      ai_total_seconds: aiTotalSeconds
    };
    writeLocalUiState(gameId, payload);
    void fetch(getApiUrl("/api/ui/state"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    }).catch(() => {});
  }, [
    aiLastMoveSeconds,
    aiTotalSeconds,
    eventFeed,
    gameId,
    hasStarted,
    aiReason,
    lastResult,
    playerLastMoveSeconds,
    playerTotalSeconds,
    statusText
  ]);

  useEffect(() => {
    const ws = new WebSocket(getWebSocketUrl("/ws/events"));
    ws.onopen = () => {
      ws.send("ready");
    };
    ws.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data);
        setEventFeed((prev) => [payload, ...prev].slice(0, 20));
      } catch {
        setEventFeed((prev) => [{ event: "raw", data: event.data }, ...prev].slice(0, 20));
      }
      ws.send("ack");
    };
    return () => {
      ws.close();
    };
  }, []);

  async function handleReset(startAfterReset = false) {
    setIsLoading(true);
    setError("");
    setWarning("");
    setPendingAnalysis(null);
    setVisionBypassPayload(null);
    let hasSettled = false;
    const resetWatchdog = window.setTimeout(() => {
      if (!hasSettled) {
        setIsLoading(false);
        setError("Reset request is taking too long. Check backend is running on port 8000.");
      }
    }, 10000);
    try {
      const response = await fetch(getApiUrl("/api/reset"), {
        method: "POST"
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error("Reset failed");
      }
      const nextFen = String(data.current_fen || currentFen);
      const nextGameId = String(data.game_id || "");
      setBoardPosition(piecePlacementToBoardPosition(fenToPiecePlacement(nextFen)));
      setCurrentFen(nextFen);
      setGameId(nextGameId);
      setHasStarted(startAfterReset);
      setStatusText(startAfterReset ? "Player move" : "Awaiting game start");
      setPlayerMoveStartedAt(startAfterReset ? Date.now() : null);
      setAiMoveStartedAt(null);
      setClockNowMs(Date.now());
      setPlayerLastMoveSeconds(0);
      setAiLastMoveSeconds(0);
      setPlayerTotalSeconds(0);
      setAiTotalSeconds(0);
      setAiReason("");
      setLastResult(null);
      setEventFeed([]);
      setPendingAnalysis(null);
      hasSettled = true;
    } catch (err) {
      hasSettled = true;
      setError(getErrorMessage(err));
    } finally {
      window.clearTimeout(resetWatchdog);
      setIsLoading(false);
    }
  }

  function applyAnalysisSuccess(data, aiStartMs) {
    const aiMoveSeconds = Math.max(0, (Date.now() - aiStartMs) / 1000);
    setAiLastMoveSeconds(aiMoveSeconds);
    setAiTotalSeconds((prev) => prev + aiMoveSeconds);
    setAiMoveStartedAt(null);
    setAiReason(String(data.ai_reason || ""));
    setLastResult(data);
    setVisionBypassPayload(null);
    if (data.post_fen) {
      const postFen = String(data.post_fen);
      setCurrentFen(postFen);
      setBoardPosition(piecePlacementToBoardPosition(fenToPiecePlacement(postFen)));
      const postBoard = new Chess(postFen);
      if (postBoard.isGameOver()) {
        setStatusText("Game over");
        setPlayerMoveStartedAt(null);
        return;
      }
    }
    setStatusText("Player move");
    setPlayerMoveStartedAt(Date.now());
    setClockNowMs(Date.now());
  }

  async function handleAnalyse(observedPiecePlacement = piecePlacement, playerTimeSeconds = playerElapsedSeconds) {
    if (!hasStarted || statusText !== "Player move") {
      return;
    }
    setIsLoading(true);
    setError("");
    setWarning("");
    setVisionBypassPayload(null);
    const aiStartMs = Date.now();
    const clampedPlayerTime = Math.max(0, Number(playerTimeSeconds || 0));
    setPlayerLastMoveSeconds(clampedPlayerTime);
    setPlayerTotalSeconds((prev) => prev + clampedPlayerTime);
    try {
      setStatusText("AI move");
      setPlayerMoveStartedAt(null);
      setAiMoveStartedAt(aiStartMs);
      let analysisImageDataUrl = null;
      if (cameraInputMode === "ui_render") {
        await waitMs(120);
        let snapshot = null;
        for (let attempt = 0; attempt < 12; attempt += 1) {
          const canvas = uiRenderFrameRef.current?.querySelector("canvas");
          if (canvas && typeof canvas.toDataURL === "function") {
            snapshot = canvas.toDataURL("image/jpeg", 0.92);
            if (snapshot && snapshot.length > 32) {
              break;
            }
          }
          await new Promise((resolve) => {
            window.requestAnimationFrame(() => resolve());
          });
        }
        if (!snapshot) {
          throw new Error("UI render camera is unavailable for snapshot capture.");
        }
        analysisImageDataUrl = snapshot;
      }
      const response = await fetch(getApiUrl("/api/player/analyse"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          player_time_s: playerTimeSeconds,
          analysis_image_data_url: analysisImageDataUrl,
          ground_truth_piece_placement: observedPiecePlacement,
          bypass_vision_with_ground_truth: false,
          view_mode: viewMode,
          camera_pitch_deg: cameraPitchDeg,
          camera_distance: cameraDistance
        })
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(data.detail || "Analysis failed");
      }
      if (data.status === "vision_mismatch_error") {
        const mismatchMessage = String(data.message || "Vision prediction did not match the player move.");
        setVisionBypassPayload({
          playerTimeSeconds: clampedPlayerTime,
          analysisImageDataUrl,
          groundTruthPiecePlacement: observedPiecePlacement,
          viewMode,
          cameraPitchDeg,
          cameraDistance,
          mismatchMessage,
          predictedMoveSan: data.predicted_move_san,
          predictedPiecePlacement: data.predicted_piece_placement,
          predictions: data.predictions || []
        });
        throw new Error(mismatchMessage);
      }
      if (data.status === "illegal_transition_warning") {
        throw new Error(data.warning || "Could not infer a legal move from the board.");
      }
      applyAnalysisSuccess(data, aiStartMs);
    } catch (err) {
      setAiMoveStartedAt(null);
      setStatusText("Player move");
      setPlayerMoveStartedAt(Date.now());
      setError(getErrorMessage(err));
    } finally {
      setIsLoading(false);
    }
  }

  async function handleBypassVision() {
    if (!visionBypassPayload || !hasStarted || statusText !== "Player move") {
      return;
    }
    const aiStartMs = Date.now();
    setIsLoading(true);
    setError("");
    setWarning("");
    try {
      setStatusText("AI move");
      setPlayerMoveStartedAt(null);
      setAiMoveStartedAt(aiStartMs);
      const response = await fetch(getApiUrl("/api/player/analyse"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          player_time_s: visionBypassPayload.playerTimeSeconds,
          analysis_image_data_url: visionBypassPayload.analysisImageDataUrl,
          ground_truth_piece_placement: visionBypassPayload.groundTruthPiecePlacement,
          bypass_vision_with_ground_truth: true,
          view_mode: visionBypassPayload.viewMode,
          camera_pitch_deg: visionBypassPayload.cameraPitchDeg,
          camera_distance: visionBypassPayload.cameraDistance
        })
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(data.detail || "Bypass analysis failed");
      }
      if (data.status === "illegal_transition_warning") {
        throw new Error(data.warning || "Could not infer a legal move from the board.");
      }
      applyAnalysisSuccess(data, aiStartMs);
    } catch (err) {
      setAiMoveStartedAt(null);
      setStatusText("Player move");
      setPlayerMoveStartedAt(Date.now());
      setError(getErrorMessage(err));
    } finally {
      setIsLoading(false);
    }
  }

  useEffect(() => {
    if (!pendingAnalysis || isLoading) {
      return;
    }
    void handleAnalyse(
      pendingAnalysis.observedPiecePlacement,
      pendingAnalysis.playerTimeSeconds
    );
    setPendingAnalysis(null);
  }, [pendingAnalysis, isLoading]);

  function canDragPiece(pieceCode) {
    return String(pieceCode || "").startsWith("w");
  }

  function onPieceDrop(sourceSquare, targetSquare, pieceCode) {
    if (isLoading || (statusText !== "Player move" && statusText !== "Awaiting game start")) {
      return false;
    }
    if (!canDragPiece(pieceCode)) {
      return false;
    }
    if (!sourceSquare || !targetSquare) {
      return false;
    }

    const board = new Chess(currentFen);
    const move = board.move({
      from: sourceSquare,
      to: targetSquare,
      promotion: "q"
    });
    if (!move) {
      return false;
    }

    const nextFen = board.fen();
    const nextPiecePlacement = fenToPiecePlacement(nextFen);
    const isFirstMoveFromAwaiting = !hasStarted && statusText === "Awaiting game start";
    if (isFirstMoveFromAwaiting) {
      setHasStarted(true);
      setStatusText("Player move");
      setPlayerMoveStartedAt(null);
      setClockNowMs(Date.now());
    }
    setCurrentFen(nextFen);
    setBoardPosition(piecePlacementToBoardPosition(nextPiecePlacement));
    setPendingAnalysis({
      observedPiecePlacement: nextPiecePlacement,
      playerTimeSeconds: isFirstMoveFromAwaiting ? 0 : playerElapsedSeconds
    });
    return true;
  }

  async function handleStartGame() {
    await handleReset(true);
  }

  return (
    <main className="app-shell">
      <section className="board-panel">
        <h1>AI Chess Agent</h1>
        <p className="subheading">Drag and drop your move.</p>
        {viewMode === "2d" ? (
          <div ref={boardFrameRef} className="board-frame">
              <Chessboard
                id="orchestrator-board"
                position={boardPosition}
                onPieceDrop={onPieceDrop}
                boardWidth={560}
                arePiecesDraggable={canInteract}
                boardOrientation="white"
                isDraggablePiece={({ piece }) => canDragPiece(piece)}
              />
          </div>
        ) : (
          <>
            <div ref={boardFrameRef} className="board-frame board-frame-3d">
              <ChessBoard3D
                boardPosition={boardPosition}
                width={560}
                cameraPitchDeg={cameraPitchDeg}
                cameraDistance={cameraDistance}
              />
            </div>
          </>
        )}
        <div ref={uiRenderFrameRef} className="hidden-camera-render" aria-hidden="true">
          <ChessBoard3D
            boardPosition={boardPosition}
            width={560}
            cameraPitchDeg={cameraPitchDeg}
            cameraDistance={cameraDistance}
          />
        </div>
      </section>

      <section className="control-panel">
        <div className="card">
          <div className="status-actions" role="group" aria-label="Game and view actions">
            <button
              type="button"
              onClick={hasStarted ? () => handleReset(false) : handleStartGame}
              disabled={isLoading}
            >
              {hasStarted ? "Reset Game" : "Start Game"}
            </button>
            <button
              type="button"
              className={viewMode === "2d" ? "toggle-btn active" : "toggle-btn"}
              onClick={() => setViewMode("2d")}
            >
              Player View
            </button>
            <button
              type="button"
              className={viewMode === "3d" ? "toggle-btn active" : "toggle-btn"}
              onClick={() => setViewMode("3d")}
            >
              Chess Camera
            </button>
          </div>
          <div className="status-row">
            <span className="status-label">Status:</span>
            <span className={`status-chip ${statusTone}`}>{statusText}</span>
          </div>
          <div className="timers-grid">
            <div className="timer-block">
              <p>Player timer: {playerTimerText}</p>
              <p>Player last move: {playerLastMoveText}</p>
              <p>Player total: {playerTotalText}</p>
            </div>
            <div className="timer-block">
              <p>AI timer: {aiTimerText}</p>
              <p>AI last move: {aiLastMoveText}</p>
              <p>AI total: {aiTotalText}</p>
            </div>
          </div>
          {warning ? <p className="warning">{warning}</p> : null}
          {error ? <p className="error">{error}</p> : null}
          {visionBypassPayload ? (
            <button type="button" onClick={handleBypassVision} disabled={isLoading}>
              Bypass Vision And Continue
            </button>
          ) : null}
        </div>

        <div className="card move-helper">
          <label className="angle-control" htmlFor="camera-angle">
            Camera Angle: {cameraPitchDeg}°
          </label>
          <input
            id="camera-angle"
            className="angle-slider"
            type="range"
            min={CAMERA_MIN_PITCH_DEG}
            max={CAMERA_MAX_PITCH_DEG}
            step="1"
            value={cameraPitchDeg}
            onChange={(event) => setCameraPitchDeg(Number(event.target.value))}
          />
          <label className="angle-control" htmlFor="camera-zoom">
            Zoom: {cameraDistance.toFixed(1)}
          </label>
          <input
            id="camera-zoom"
            className="angle-slider"
            type="range"
            min="10"
            max="24"
            step="0.5"
            value={cameraDistance}
            onChange={(event) => setCameraDistance(Number(event.target.value))}
          />
        </div>

        <div className="card">
          <h2>AI Reason</h2>
          <pre>{aiReason || "No AI decision yet."}</pre>
        </div>

        <div className="card">
          <h2>Events</h2>
          <pre>{JSON.stringify(eventFeed, null, 2)}</pre>
        </div>
      </section>
    </main>
  );
}
