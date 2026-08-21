"use client";

import React, { useRef, useState, useEffect, useMemo } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Text } from "@react-three/drei";
import * as THREE from "three";
import {
  Brain,
  RotateCw,
  ZoomIn,
  ShieldCheck,
  Lock,
  Crosshair,
  Sparkles,
  Activity,
} from "lucide-react";

export type BrainViewerProps = {
  prediction?: string;
  confidence?: number;
};

// ----------------------------------------------------------------------
// Procedural Anatomical Brain Geometry with Real Gyri & Sulci Folds
// ----------------------------------------------------------------------
function createAnatomicalHemisphereGeometry(isLeft: boolean) {
  const geom = new THREE.SphereGeometry(0.72, 48, 48);
  const pos = geom.attributes.position;
  const vec = new THREE.Vector3();

  for (let i = 0; i < pos.count; i++) {
    vec.fromBufferAttribute(pos, i);

    if (isLeft && vec.x > -0.05) {
      vec.x = -0.05 + (vec.x + 0.05) * 0.2;
    } else if (!isLeft && vec.x < 0.05) {
      vec.x = 0.05 + (vec.x - 0.05) * 0.2;
    }

    if (vec.z > 0.2) vec.z *= 1.15;
    if (vec.z < -0.2) vec.z *= 1.1;
    if (vec.y < 0 && Math.abs(vec.x) > 0.2) vec.y *= 0.85;

    const frequency1 = 8.5;
    const frequency2 = 16.0;
    const noise1 =
      Math.sin(vec.x * frequency1) *
      Math.cos(vec.y * frequency1) *
      Math.sin(vec.z * frequency1);
    const noise2 =
      Math.cos(vec.x * frequency2 + vec.y * frequency2) *
      Math.sin(vec.z * frequency2);

    const displacement = 0.045 * noise1 + 0.02 * noise2;
    vec.addScaledVector(vec.clone().normalize(), displacement);

    pos.setXYZ(i, vec.x, vec.y, vec.z);
  }

  geom.computeVertexNormals();
  return geom;
}

function createCerebellumGeometry() {
  const geom = new THREE.SphereGeometry(0.48, 36, 36);
  const pos = geom.attributes.position;
  const vec = new THREE.Vector3();

  for (let i = 0; i < pos.count; i++) {
    vec.fromBufferAttribute(pos, i);
    vec.y *= 0.7;
    vec.z *= 0.85;
    const folia = Math.sin(vec.y * 22.0) * 0.025;
    vec.addScaledVector(vec.clone().normalize(), folia);
    pos.setXYZ(i, vec.x, vec.y, vec.z);
  }

  geom.computeVertexNormals();
  return geom;
}

// ----------------------------------------------------------------------
// 1. Anatomical Holographic Brain Mesh with Imperceptible Camera Drift
// ----------------------------------------------------------------------
function AnatomicalBrain({
  progress,
  confidence = 95.06,
}: {
  progress: number;
  confidence: number;
}) {
  const groupRef = useRef<THREE.Group>(null);
  const tumorRef = useRef<THREE.Group>(null);

  const leftHemisphereGeom = useMemo(() => createAnatomicalHemisphereGeometry(true), []);
  const rightHemisphereGeom = useMemo(() => createAnatomicalHemisphereGeometry(false), []);
  const cerebellumGeom = useMemo(() => createCerebellumGeometry(), []);

  // Materialization stages:
  // 0.0 - 0.5s: Particles converge
  // 0.5 - 0.8s: Gyri Wireframe appears (strictly under 0.8s)
  // 0.8 - 1.5s: Fills into glass body with edge glow
  // 1.5 - 2.0s: Organic tumor lights up with confidence-scaled emission
  const wireframeOpacity =
    progress < 0.5
      ? 0
      : progress <= 0.8
      ? (progress - 0.5) / 0.3
      : Math.max(0.12, 1 - (progress - 0.8) / 0.7);

  const glassOpacity =
    progress < 0.8 ? 0 : Math.min(0.42, (progress - 0.8) / 0.7);

  const tumorProgress =
    progress < 1.5 ? 0 : Math.min(1, (progress - 1.5) / 0.5);

  const confidenceScale = Math.max(0.65, Math.min(1.2, confidence / 95));
  const baseEmissiveIntensity = 2.2 * confidenceScale;

  useFrame((state, delta) => {
    if (groupRef.current) {
      groupRef.current.rotation.y += delta * 0.11;
      // Clinical imperceptible vertical camera/group drift (Siemens/Philips style)
      groupRef.current.position.y = 0.02 * Math.sin(state.clock.getElapsedTime() * 0.25);
    }

    if (tumorRef.current && tumorProgress > 0) {
      const pulse = 1 + 0.15 * Math.sin(state.clock.getElapsedTime() * 2.5);
      tumorRef.current.scale.setScalar(1 + 0.02 * Math.sin(state.clock.getElapsedTime() * 3.0));

      tumorRef.current.children.forEach((child) => {
        if ((child as THREE.Mesh).material) {
          const mat = (child as THREE.Mesh).material as THREE.MeshStandardMaterial;
          mat.emissiveIntensity = tumorProgress * baseEmissiveIntensity * pulse;
        }
      });
    }
  });

  return (
    <group ref={groupRef}>
      {/* LEFT CEREBRAL HEMISPHERE */}
      <group position={[-0.38, 0.1, 0]}>
        <mesh geometry={leftHemisphereGeom}>
          <meshStandardMaterial
            color="#22D3EE"
            wireframe={true}
            transparent={true}
            opacity={wireframeOpacity * 0.5}
          />
        </mesh>
        <mesh geometry={leftHemisphereGeom}>
          <meshPhysicalMaterial
            color="#131A2A"
            emissive="#3B82F6"
            emissiveIntensity={0.38}
            roughness={0.12}
            metalness={0.15}
            transparent={true}
            opacity={glassOpacity}
            transmission={0.65}
            clearcoat={1.0}
            clearcoatRoughness={0.1}
            ior={1.3}
          />
        </mesh>
      </group>

      {/* RIGHT CEREBRAL HEMISPHERE */}
      <group position={[0.38, 0.1, 0]}>
        <mesh geometry={rightHemisphereGeom}>
          <meshStandardMaterial
            color="#22D3EE"
            wireframe={true}
            transparent={true}
            opacity={wireframeOpacity * 0.5}
          />
        </mesh>
        <mesh geometry={rightHemisphereGeom}>
          <meshPhysicalMaterial
            color="#131A2A"
            emissive="#3B82F6"
            emissiveIntensity={0.38}
            roughness={0.12}
            metalness={0.15}
            transparent={true}
            opacity={glassOpacity}
            transmission={0.65}
            clearcoat={1.0}
            clearcoatRoughness={0.1}
            ior={1.3}
          />
        </mesh>

        {/* ORGANIC TUMOR UPGRADE (Embedded inside Right Hemisphere) */}
        <group ref={tumorRef} position={[0.18, 0.12, 0.12]}>
          <mesh position={[0, 0, 0]}>
            <sphereGeometry args={[0.22, 24, 24]} />
            <meshStandardMaterial
              color="#EF4444"
              emissive="#EF4444"
              emissiveIntensity={tumorProgress * baseEmissiveIntensity}
              transparent={true}
              opacity={tumorProgress * 0.92}
              roughness={0.2}
            />
          </mesh>
          <mesh position={[0.1, 0.07, -0.05]}>
            <sphereGeometry args={[0.14, 20, 20]} />
            <meshStandardMaterial
              color="#EF4444"
              emissive="#EF4444"
              emissiveIntensity={tumorProgress * baseEmissiveIntensity * 0.85}
              transparent={true}
              opacity={tumorProgress * 0.85}
            />
          </mesh>
          <mesh position={[-0.07, -0.09, 0.07]}>
            <sphereGeometry args={[0.12, 20, 20]} />
            <meshStandardMaterial
              color="#EF4444"
              emissive="#EF4444"
              emissiveIntensity={tumorProgress * baseEmissiveIntensity * 0.8}
              transparent={true}
              opacity={tumorProgress * 0.8}
            />
          </mesh>
        </group>
      </group>

      {/* CEREBELLUM */}
      <group position={[0, -0.48, -0.36]}>
        <mesh geometry={cerebellumGeom}>
          <meshStandardMaterial
            color="#22D3EE"
            wireframe={true}
            transparent={true}
            opacity={wireframeOpacity * 0.4}
          />
        </mesh>
        <mesh geometry={cerebellumGeom}>
          <meshPhysicalMaterial
            color="#131A2A"
            emissive="#3B82F6"
            emissiveIntensity={0.28}
            transparent={true}
            opacity={glassOpacity * 0.85}
          />
        </mesh>
      </group>

      {/* BRAINSTEM */}
      <group position={[0, -0.78, -0.1]} rotation={[0.15, 0, 0]}>
        <mesh>
          <cylinderGeometry args={[0.16, 0.12, 0.55, 20]} />
          <meshStandardMaterial
            color="#22D3EE"
            wireframe={true}
            transparent={true}
            opacity={wireframeOpacity * 0.35}
          />
        </mesh>
        <mesh>
          <cylinderGeometry args={[0.15, 0.11, 0.53, 20]} />
          <meshPhysicalMaterial
            color="#131A2A"
            emissive="#3B82F6"
            emissiveIntensity={0.22}
            transparent={true}
            opacity={glassOpacity * 0.8}
          />
        </mesh>
      </group>
    </group>
  );
}

// ----------------------------------------------------------------------
// 2. Volumetric Scan Sweep (Subtle 7s Left-to-Right Cyan Laser Plane)
// ----------------------------------------------------------------------
function VolumetricScanBeam({ progress }: { progress: number }) {
  const beamRef = useRef<THREE.Mesh>(null);

  useFrame((state) => {
    if (beamRef.current && progress >= 0.8) {
      const cycleTime = 7.0;
      const t = (state.clock.getElapsedTime() % cycleTime) / cycleTime;
      beamRef.current.position.x = -1.4 + t * 2.8;

      const fade = Math.sin(t * Math.PI);
      if (beamRef.current.material) {
        (beamRef.current.material as THREE.MeshBasicMaterial).opacity = 0.22 * fade;
      }
    }
  });

  if (progress < 0.8) return null;

  return (
    <mesh ref={beamRef} rotation={[0, Math.PI / 2, 0]}>
      <planeGeometry args={[2.4, 2.4]} />
      <meshBasicMaterial
        color="#22D3EE"
        transparent={true}
        opacity={0.2}
        side={THREE.DoubleSide}
        blending={THREE.AdditiveBlending}
      />
    </mesh>
  );
}

// ----------------------------------------------------------------------
// 3. Confidence-Scaled Holographic Pulse Ring (Non-Obscuring Thin Ring)
// ----------------------------------------------------------------------
function ConfidencePulseRing({
  progress,
  confidence = 95.06,
}: {
  progress: number;
  confidence: number;
}) {
  const ringRef = useRef<THREE.Mesh>(null);

  const confidenceFactor = Math.max(0.7, Math.min(1.25, confidence / 95));
  const speed = 1.2 * confidenceFactor;

  useFrame((state) => {
    if (ringRef.current && progress >= 0.9) {
      const t = (state.clock.getElapsedTime() * speed) % 2.0;
      const expansion = t / 2.0;
      const scale = 1.0 + expansion * 0.4;
      ringRef.current.scale.setScalar(scale);

      if (ringRef.current.material) {
        const opacity = (1 - expansion) * 0.35 * confidenceFactor;
        (ringRef.current.material as THREE.MeshBasicMaterial).opacity = opacity;
      }
    }
  });

  if (progress < 0.9) return null;

  return (
    <mesh ref={ringRef} position={[0.56, 0.22, 0.12]} rotation={[Math.PI / 4, 0, 0]}>
      <torusGeometry args={[0.32, 0.005, 16, 64]} />
      <meshBasicMaterial
        color="#EF4444"
        transparent={true}
        opacity={0.35}
        blending={THREE.AdditiveBlending}
      />
    </mesh>
  );
}

// ----------------------------------------------------------------------
// 4. Rotating Coordinate Ring (0°, 90°, 180°, 270° Cardinal Markers)
// ----------------------------------------------------------------------
function CoordinateRing({ progress }: { progress: number }) {
  const ringRef = useRef<THREE.Group>(null);
  const opacity = Math.min(0.4, Math.max(0, (progress - 1.0) / 0.8));

  useFrame((_, delta) => {
    if (ringRef.current) {
      ringRef.current.rotation.y += delta * 0.08;
    }
  });

  return (
    <group ref={ringRef} position={[0, -0.05, 0]}>
      <mesh rotation={[Math.PI / 2, 0, 0]}>
        <torusGeometry args={[2.5, 0.004, 16, 120]} />
        <meshBasicMaterial color="#22D3EE" transparent={true} opacity={opacity} />
      </mesh>

      {progress >= 0.9 && (
        <group>
          <Text position={[2.6, 0, 0]} fontSize={0.08} color="#22D3EE" fillOpacity={opacity * 0.8}>
            0°
          </Text>
          <Text position={[0, 0, 2.6]} fontSize={0.08} color="#22D3EE" fillOpacity={opacity * 0.8}>
            90°
          </Text>
          <Text position={[-2.6, 0, 0]} fontSize={0.08} color="#22D3EE" fillOpacity={opacity * 0.8}>
            180°
          </Text>
          <Text position={[0, 0, -2.6]} fontSize={0.08} color="#22D3EE" fillOpacity={opacity * 0.8}>
            270°
          </Text>
        </group>
      )}
    </group>
  );
}

// ----------------------------------------------------------------------
// 5. Floating Diagnostic Data Labels (Subtle <25% Opacity Tags)
// ----------------------------------------------------------------------
function FloatingDataLabels({
  progress,
  confidence = 95.06,
}: {
  progress: number;
  confidence: number;
}) {
  const groupRef = useRef<THREE.Group>(null);
  const fillOpacity = progress >= 1.0 ? 0.22 : 0;

  useFrame((state) => {
    if (groupRef.current) {
      groupRef.current.position.y = 0.02 * Math.sin(state.clock.getElapsedTime() * 0.8);
    }
  });

  if (progress < 1.0) return null;

  return (
    <group ref={groupRef}>
      <Text position={[-1.8, 1.3, 0]} fontSize={0.075} color="#22D3EE" fillOpacity={fillOpacity}>
        SCAN READY
      </Text>
      <Text position={[1.7, 1.3, 0]} fontSize={0.075} color="#3B82F6" fillOpacity={fillOpacity}>
        AI LOCKED
      </Text>
      <Text position={[-1.8, -1.3, 0]} fontSize={0.075} color="#22D3EE" fillOpacity={fillOpacity}>
        VOL RECON
      </Text>
      <Text position={[1.6, -1.3, 0]} fontSize={0.075} color="#22C55E" fillOpacity={fillOpacity}>
        CONF: {confidence.toFixed(1)}%
      </Text>
    </group>
  );
}

// ----------------------------------------------------------------------
// 6. Converging Swarm Particles
// ----------------------------------------------------------------------
function ParticleSwarm({ progress }: { progress: number }) {
  const pointsRef = useRef<THREE.Points>(null);
  const particleCount = 160;

  const positions = useMemo(() => {
    const pos = new Float32Array(particleCount * 3);
    for (let i = 0; i < particleCount; i++) {
      const radius = 2.2 + Math.random() * 1.5;
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);
      pos[i * 3] = radius * Math.sin(phi) * Math.cos(theta);
      pos[i * 3 + 1] = radius * Math.sin(phi) * Math.sin(theta);
      pos[i * 3 + 2] = radius * Math.cos(phi);
    }
    return pos;
  }, []);

  const particleOpacity = Math.min(0.75, progress * 0.85);

  useFrame((_, delta) => {
    if (pointsRef.current) {
      pointsRef.current.rotation.y += delta * 0.08;
      pointsRef.current.rotation.x += delta * 0.035;
    }
  });

  return (
    <points ref={pointsRef}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[positions, 3]} />
      </bufferGeometry>
      <pointsMaterial
        size={0.03}
        color="#22D3EE"
        transparent={true}
        opacity={particleOpacity}
        blending={THREE.AdditiveBlending}
      />
    </points>
  );
}

// ----------------------------------------------------------------------
// 7. Main Exported BrainViewer Component
// ----------------------------------------------------------------------
export default function BrainViewer({
  prediction = "Glioma",
  confidence = 98.4,
}: BrainViewerProps) {
  const [mounted, setMounted] = useState(false);
  const [progress, setProgress] = useState(0);
  const [autoRotate, setAutoRotate] = useState(true);
  const [showFlash, setShowFlash] = useState(false);
  const timerRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    setMounted(true);
  }, []);

  // 2-Second Materialization Sequence & Single 120ms HUD Camera Flash Trigger
  useEffect(() => {
    if (!mounted) return;
    const startTime = Date.now();
    const duration = 2000;

    const interval = setInterval(() => {
      const elapsed = Date.now() - startTime;
      const currentProgress = Math.min(1, elapsed / duration);
      setProgress(currentProgress);

      if (currentProgress >= 1) {
        clearInterval(interval);
        setShowFlash(true);
        setTimeout(() => setShowFlash(false), 120);
      }
    }, 16);

    return () => clearInterval(interval);
  }, [mounted]);

  const handleControlStart = () => {
    setAutoRotate(false);
    if (timerRef.current) clearTimeout(timerRef.current);
  };

  const handleControlEnd = () => {
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = setTimeout(() => {
      setAutoRotate(true);
    }, 3000);
  };

  if (!mounted) {
    return (
      <div className="w-full aspect-square rounded-3xl bg-[#131A2A] border border-[#263245] flex items-center justify-center font-mono text-xs text-[#94A3B8]">
        INITIALIZING 3D ANATOMICAL ENGINE...
      </div>
    );
  }

  return (
    <div className="w-full bg-[#131A2A] border border-[#263245] rounded-3xl p-5 shadow-2xl relative overflow-hidden flex flex-col gap-4 font-sans">
      {/* 120ms Single HUD Camera Reconstruction Flash Overlay */}
      {showFlash && (
        <div className="absolute inset-0 bg-[#3B82F6]/30 backdrop-brightness-150 z-50 pointer-events-none transition-opacity duration-120 animate-pulse" />
      )}

      {/* Corner Targeting Brackets */}
      <div className="absolute top-2 left-2 w-4 h-4 border-t-2 border-l-2 border-[#3B82F6]/60 pointer-events-none z-20" />
      <div className="absolute top-2 right-2 w-4 h-4 border-t-2 border-r-2 border-[#3B82F6]/60 pointer-events-none z-20" />
      <div className="absolute bottom-2 left-2 w-4 h-4 border-b-2 border-l-2 border-[#3B82F6]/60 pointer-events-none z-20" />
      <div className="absolute bottom-2 right-2 w-4 h-4 border-b-2 border-r-2 border-[#3B82F6]/60 pointer-events-none z-20" />

      {/* Header Bar */}
      <div className="flex items-center justify-between border-b border-[#263245] pb-3.5 relative z-10">
        <div className="flex items-center gap-2.5">
          <div className="p-2 rounded-xl bg-[#1A2336] border border-[#263245] text-[#3B82F6]">
            <Brain className="w-4 h-4 text-[#3B82F6]" />
          </div>
          <div>
            <h3 className="text-xs font-mono font-bold uppercase tracking-wider text-[#F8FAFC]">
              3D BRAIN VOLUME RECONSTRUCTION
            </h3>
            <p className="text-[11px] text-[#94A3B8]">
              AI Tissue Visualization • ResNet-50
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <span className="text-[10px] font-mono text-[#22D3EE] bg-[#1A2336] border border-[#263245] px-2.5 py-1 rounded flex items-center gap-1.5">
            <Crosshair className="w-3 h-3 text-[#22D3EE] animate-pulse" />
            ANATOMICAL HUD
          </span>
        </div>
      </div>

      {/* 3D WebGL Canvas Container */}
      <div className="relative w-full aspect-square sm:aspect-[4/3] rounded-2xl bg-[#0A0F1C] border border-[#263245] overflow-hidden group">
        {/* Cyan Scan Ticks Overlay */}
        <div className="absolute top-2 left-1/2 -translate-x-1/2 flex items-center gap-1 text-[9px] font-mono text-[#22D3EE]/40 pointer-events-none z-10">
          <span>+</span>
          <span>- - - 3D VOLUMETRIC GRID - - -</span>
          <span>+</span>
        </div>

        {/* Materialization Sequence HUD Status Pill */}
        <div className="absolute top-3 left-3 z-10 flex items-center gap-2 bg-[#131A2A]/90 border border-[#263245] px-3 py-1.5 rounded-lg backdrop-blur-md text-[11px] font-mono text-[#F8FAFC]">
          <span className="w-2 h-2 rounded-full bg-[#22C55E] animate-pulse" />
          <span>
            {progress < 1
              ? `RECONSTRUCTING... ${Math.round(progress * 100)}%`
              : "ANATOMICAL 3D ACTIVE"}
          </span>
        </div>

        {/* Prediction Target Highlight Tag */}
        {progress >= 0.9 && (
          <div className="absolute bottom-3 left-3 z-10 flex items-center gap-2 bg-[#131A2A]/90 border border-[#EF4444]/40 px-3 py-1.5 rounded-lg backdrop-blur-md text-[11px] font-mono">
            <span className="w-2 h-2 rounded-full bg-[#EF4444] animate-ping" />
            <span className="text-[#EF4444] font-bold uppercase">
              TUMOR LOCATED: {prediction} ({confidence}%)
            </span>
          </div>
        )}

        {/* Three.js Canvas */}
        <Canvas
          camera={{ position: [0, 0, 4.5], fov: 45 }}
          className="w-full h-full cursor-grab active:cursor-grabbing"
        >
          <ambientLight intensity={0.65} />
          <directionalLight position={[10, 10, 10]} intensity={1.3} />
          <pointLight position={[-10, -10, -10]} color="#3B82F6" intensity={0.9} />
          <pointLight position={[2, 1, 2]} color="#EF4444" intensity={1.8} />

          <AnatomicalBrain progress={progress} confidence={confidence} />
          <VolumetricScanBeam progress={progress} />
          <ConfidencePulseRing progress={progress} confidence={confidence} />
          <CoordinateRing progress={progress} />
          <FloatingDataLabels progress={progress} confidence={confidence} />
          <ParticleSwarm progress={progress} />

          <OrbitControls
            enablePan={false}
            enableZoom={true}
            minDistance={3.0}
            maxDistance={7.5}
            rotateSpeed={0.8}
            zoomSpeed={0.8}
            dampingFactor={0.05}
            autoRotate={autoRotate && progress >= 0.9}
            autoRotateSpeed={1.4}
            onStart={handleControlStart}
            onEnd={handleControlEnd}
          />
        </Canvas>
      </div>

      {/* Footer Navigation & Controls Guide */}
      <div className="flex flex-wrap items-center justify-between gap-2 text-[11px] font-mono text-[#94A3B8] border-t border-[#263245] pt-3 relative z-10">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5">
            <RotateCw className="w-3.5 h-3.5 text-[#3B82F6]" />
            <span>DRAG ROTATE</span>
          </div>
          <div className="flex items-center gap-1.5">
            <ZoomIn className="w-3.5 h-3.5 text-[#22D3EE]" />
            <span>SCROLL ZOOM</span>
          </div>
        </div>

        <div className="flex items-center gap-1.5 text-[#22C55E] bg-[#22C55E]/10 border border-[#22C55E]/20 px-2 py-0.5 rounded">
          <Lock className="w-3 h-3 text-[#22C55E]" />
          <span>AI LOCKED</span>
        </div>
      </div>
    </div>
  );
}
