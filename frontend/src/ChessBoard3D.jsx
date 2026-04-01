import { Suspense, useMemo } from "react";
import * as THREE from "three";
import { Canvas, useThree } from "@react-three/fiber";
import { OrbitControls, Text, useGLTF } from "@react-three/drei";

const MODEL_PATH = "/models/chess/low_poly_chess_set.glb";
const PIECE_Y_OFFSET = 0.50;
const LABEL_Y_OFFSET = 0.88;
const FILES = ["a", "b", "c", "d", "e", "f", "g", "h"];
const RANKS = [8, 7, 6, 5, 4, 3, 2, 1];

function boardPositionToPieces(boardPosition) {
  const pieces = [];
  for (let rankIdx = 0; rankIdx < RANKS.length; rankIdx += 1) {
    const rank = RANKS[rankIdx];
    for (let fileIdx = 0; fileIdx < FILES.length; fileIdx += 1) {
      const square = `${FILES[fileIdx]}${rank}`;
      const pieceCode = boardPosition[square];
      if (!pieceCode) {
        continue;
      }
      pieces.push({
        pieceCode,
        x: fileIdx - 3.5,
        z: rankIdx - 3.5
      });
    }
  }
  return pieces;
}

function pieceYawFromCode(pieceCode) {
  const role = String(pieceCode || "").slice(1);
  if (role === "N" || role === "B") {
    return Math.PI / 2;
  }
  return 0;
}

function pieceLetterFromName(name) {
  const lower = String(name || "").toLowerCase();
  if (lower.includes("pawn")) {
    return "P";
  }
  if (lower.includes("knight")) {
    return "N";
  }
  if (lower.includes("bishop")) {
    return "B";
  }
  if (lower.includes("rook")) {
    return "R";
  }
  if (lower.includes("queen")) {
    return "Q";
  }
  if (lower.includes("king")) {
    return "K";
  }
  return null;
}

function isDarkPiece(mesh) {
  const meshName = String(mesh.name || "").toLowerCase();
  const materialName = String(mesh.material?.name || "").toLowerCase();
  return meshName.includes("dark") || materialName.includes("dark");
}

function buildPieceTemplates(scene) {
  scene.updateMatrixWorld(true);
  const templatesByCode = {};

  scene.traverse((node) => {
    if (!node.isMesh || !node.geometry) {
      return;
    }
    if (String(node.name || "").toLowerCase().includes("board")) {
      return;
    }

    const pieceLetter = pieceLetterFromName(node.name);
    if (!pieceLetter) {
      return;
    }
    const pieceCode = `${isDarkPiece(node) ? "b" : "w"}${pieceLetter}`;
    if (templatesByCode[pieceCode]) {
      return;
    }

    const geometry = node.geometry.clone();
    geometry.applyMatrix4(node.matrixWorld);
    geometry.computeBoundingBox();
    const bbox = geometry.boundingBox;
    if (!bbox) {
      return;
    }

    const centreX = (bbox.min.x + bbox.max.x) / 2;
    const centreZ = (bbox.min.z + bbox.max.z) / 2;
    const minY = bbox.min.y;
    geometry.translate(-centreX, -minY, -centreZ);
    geometry.computeVertexNormals();
    geometry.computeBoundingSphere();

    const clonedMaterial = Array.isArray(node.material)
      ? node.material.map((material) => material.clone())
      : node.material.clone();

    const tunedMaterial = Array.isArray(clonedMaterial) ? clonedMaterial : [clonedMaterial];
    tunedMaterial.forEach((material) => {
      if (pieceCode.startsWith("b")) {
        material.color = new THREE.Color("#646c78");
        material.emissive = new THREE.Color("#394150");
        material.emissiveIntensity = 0.32;
        material.roughness = 0.64;
        material.metalness = 0.03;
      } else {
        material.color = new THREE.Color("#f0ede4");
        material.emissive = new THREE.Color("#000000");
        material.emissiveIntensity = 0.0;
        material.roughness = 0.58;
        material.metalness = 0.03;
      }
    });

    templatesByCode[pieceCode] = {
      geometry,
      material: Array.isArray(clonedMaterial) ? clonedMaterial : clonedMaterial
    };
  });

  return templatesByCode;
}

function buildBoardTemplates(scene) {
  scene.updateMatrixWorld(true);
  const boardNodes = [];
  const boardBbox = new THREE.Box3();
  let hasBoardGeometry = false;

  scene.traverse((node) => {
    if (!node.isMesh || !node.geometry) {
      return;
    }
    if (!String(node.name || "").toLowerCase().includes("board")) {
      return;
    }
    const geometry = node.geometry.clone();
    geometry.applyMatrix4(node.matrixWorld);
    geometry.computeBoundingBox();
    if (geometry.boundingBox) {
      boardBbox.expandByPoint(geometry.boundingBox.min);
      boardBbox.expandByPoint(geometry.boundingBox.max);
      hasBoardGeometry = true;
    }
    const clonedMaterial = Array.isArray(node.material)
      ? node.material.map((material) => material.clone())
      : node.material.clone();
    const tunedMaterial = Array.isArray(clonedMaterial) ? clonedMaterial : [clonedMaterial];
    tunedMaterial.forEach((material) => {
      const lowerName = String(material.name || "").toLowerCase();
      if (lowerName.includes("dark")) {
        material.color = new THREE.Color("#5f4b3a");
        material.roughness = 0.82;
        material.metalness = 0.02;
      } else if (lowerName.includes("light")) {
        material.color = new THREE.Color("#e6dfcf");
        material.roughness = 0.86;
        material.metalness = 0.01;
      }
    });

    boardNodes.push({
      geometry,
      material: clonedMaterial
    });
  });

  if (!hasBoardGeometry) {
    return [];
  }

  const centreX = (boardBbox.min.x + boardBbox.max.x) / 2;
  const centreZ = (boardBbox.min.z + boardBbox.max.z) / 2;
  const minY = boardBbox.min.y;

  return boardNodes.map((entry) => {
    entry.geometry.translate(-centreX, -minY, -centreZ);
    entry.geometry.computeVertexNormals();
    entry.geometry.computeBoundingSphere();
    return entry;
  });
}

function MissingTemplatePiece({ x, z }) {
  return (
    <group position={[x, PIECE_Y_OFFSET, z]}>
      <mesh castShadow receiveShadow>
        <cylinderGeometry args={[0.28, 0.34, 0.82, 18]} />
        <meshStandardMaterial color="#c91616" roughness={0.25} metalness={0.05} />
      </mesh>
      <mesh position={[0, 0.5, 0]} castShadow>
        <boxGeometry args={[0.24, 0.18, 0.24]} />
        <meshStandardMaterial color="#ffdd57" roughness={0.2} metalness={0.1} />
      </mesh>
    </group>
  );
}

function GLTFBoard() {
  const { scene } = useGLTF(MODEL_PATH);
  const boardTemplates = useMemo(() => buildBoardTemplates(scene), [scene]);

  return (
    <>
      {boardTemplates.map((entry, idx) => (
        <mesh
          key={`board-${idx}`}
          geometry={entry.geometry}
          material={entry.material}
          position={[0, 0, 0]}
          castShadow
          receiveShadow
        />
      ))}
    </>
  );
}

function GLTFPieces({ boardPosition }) {
  const { scene } = useGLTF(MODEL_PATH);
  const pieces = useMemo(() => boardPositionToPieces(boardPosition), [boardPosition]);
  const templatesByCode = useMemo(() => buildPieceTemplates(scene), [scene]);

  return (
    <>
      {pieces.map((piece, idx) => {
        const template = templatesByCode[piece.pieceCode];
        if (!template) {
          return <MissingTemplatePiece key={`missing-${idx}`} x={piece.x} z={piece.z} />;
        }
        const yaw = pieceYawFromCode(piece.pieceCode);
        return (
          <mesh
            key={`${piece.pieceCode}-${piece.x}-${piece.z}-${idx}`}
            geometry={template.geometry}
            material={template.material}
            position={[piece.x, PIECE_Y_OFFSET, piece.z]}
            rotation={[0, yaw, 0]}
            castShadow
            receiveShadow
          />
        );
      })}
    </>
  );
}

function CameraController({ camY, camZ, fov }) {
  const { camera } = useThree();
  camera.position.set(0, camY, camZ);
  camera.fov = fov;
  camera.near = 0.1;
  camera.far = 60;
  camera.updateProjectionMatrix();
  camera.lookAt(0, 0.45, -0.4);
  return null;
}

function BoardCoordinates() {
  return (
    <group>
      {FILES.map((file, idx) => {
        const x = idx - 3.5;
        return (
          <Text
            key={`file-${file}`}
            position={[x, LABEL_Y_OFFSET, 4.42]}
            rotation={[-Math.PI / 2, 0, 0]}
            fontSize={0.24}
            color="#ffffff"
            outlineWidth={0.004}
            outlineColor="#111111"
            anchorX="center"
            anchorY="middle"
          >
            {file}
          </Text>
        );
      })}
      {RANKS.map((rank, idx) => {
        const z = idx - 3.5;
        return (
          <Text
            key={`rank-${rank}`}
            position={[-4.42, LABEL_Y_OFFSET, z]}
            rotation={[-Math.PI / 2, 0, 0]}
            fontSize={0.24}
            color="#ffffff"
            outlineWidth={0.004}
            outlineColor="#111111"
            anchorX="center"
            anchorY="middle"
          >
            {String(rank)}
          </Text>
        );
      })}
    </group>
  );
}

export default function ChessBoard3D({
  boardPosition,
  width = 560,
  cameraPitchDeg = 37,
  cameraDistance = 15.0
}) {
  const pitchRad = (cameraPitchDeg * Math.PI) / 180;
  const camY = Math.max(4.2, Math.sin(pitchRad) * cameraDistance);
  const camZ = Math.max(0.35, Math.cos(pitchRad) * cameraDistance);
  const fov = 38;

  return (
    <div style={{ width: `${width}px`, height: `${width}px` }}>
      <Canvas
        shadows
        gl={{ preserveDrawingBuffer: true }}
        camera={{
          position: [0, camY, camZ],
          fov,
          near: 0.1,
          far: 60
        }}
      >
        <CameraController camY={camY} camZ={camZ} fov={fov} />
        <color attach="background" args={["#cfd7e8"]} />
        <ambientLight intensity={0.48} />
        <hemisphereLight intensity={0.38} color="#ffffff" groundColor="#4f5a66" />
        <pointLight
          castShadow
          intensity={1.8}
          color="#ffffff"
          position={[4.08, 5.9, 1.01]}
          shadow-mapSize-width={2048}
          shadow-mapSize-height={2048}
        />
        <pointLight intensity={0.65} color="#f8fbff" position={[-6.5, 5.2, -5.8]} />
        <pointLight intensity={0.45} color="#ffe7c0" position={[7.0, 4.2, 7.6]} />
        <directionalLight intensity={0.92} color="#bfd6ff" position={[-5.8, 4.6, -9.2]} />
        <Suspense fallback={null}>
          <GLTFBoard />
          <GLTFPieces boardPosition={boardPosition} />
          <BoardCoordinates />
        </Suspense>

        <mesh position={[0, -0.07, 0]} receiveShadow>
          <boxGeometry args={[8.8, 0.12, 8.8]} />
          <meshStandardMaterial color="#2b2b2b" roughness={0.8} metalness={0.04} />
        </mesh>

        <OrbitControls
          enablePan={false}
          enableZoom={false}
          minPolarAngle={0.04}
          maxPolarAngle={1.44}
          minAzimuthAngle={-0.4}
          maxAzimuthAngle={0.4}
          target={[0, 0.45, -0.4]}
        />
      </Canvas>
    </div>
  );
}

useGLTF.preload(MODEL_PATH);
