"use client";

import React, { useRef, useState, useEffect } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import {
  Brain,
  RotateCw,
  ZoomIn,
  Sparkles,
  Activity,
  ShieldCheck,
} from "lucide-react";

export type BrainViewerProps = {
  prediction?: string;
  confidence?: number;
};

// ----------------------------------------------------------------------
// 1. Organic Brain Mesh with Dual Wireframe & Holographic Glass Body
// ----------------------------------------------------------------------
function HolographicBrain({
  progress,
}: {
  progress: number;
}) {
  const groupRef = useRef<THREE.Group>(null);
  const tumorRef = useRef<THREE.Group>(null);

  // Materialization stages derived from 0.0 -> 1.0 progress over 2.0s
  // 0.0 -> 0.25: Particle & Wireframe setup
  // 0.25 -> 0.6: Wireframe visibility
  // 0.6 -> 0.85: Glass body fade-in
  // 0.85 -> 1.0: Tumor ignition & pulsing glow
  const wireframeOpacity = Math.min(1, Math.max(0, (progress - 0.1) / 0.4));
  const glassOpacity = Math.min(0.45, Math.max(0, (progress - 0.5) / 0.35));
  const tumorProgress = Math.min(1, Math.max(0, (progress - 0.75) / 0.25));

  useFrame((state, delta) => {
    // Gentle floating rotation animation
    if (groupRef.current) {
      groupRef.current.rotation.y += delta * 0.15;
    }

    // Tumor pulsing glow effect
    if (tumorRef.current && tumorProgress > 0) {
      const pulse = 1 + 0.25 * Math.sin(state.clock.getElapsedTime() * 3.5);
      tumorRef.current.scale.setScalar(1 + 0.04 * Math.sin(state.clock.getElapsedTime() * 4));
      
      // Update emissive intensity of tumor child meshes
      tumorRef.current.children.forEach((child) => {
        if ((child as THREE.Mesh).material) {
          const mat = (child as THREE.Mesh).material as THREE.MeshStandardMaterial;
          mat.emissiveIntensity = tumorProgress * 2.2 * pulse;
        }
      });
    }
  });

  return (
    <group ref={groupRef}>
      {/* --- CEREBRUM LEFT HEMISPHERE --- */}
      <group position={[-0.48, 0.15, 0]}>
        {/* Wireframe Shell */}
        <mesh>
          <sphereGeometry args={[0.72, 28, 28]} />
          <meshStandardMaterial
            color="#22D3EE"
            wireframe={true}
            transparent={true}
            opacity={wireframeOpacity * 0.4}
          />
        </mesh>
        {/* Holographic Translucent Glass Body */}
        <mesh>
          <sphereGeometry args={[0.7, 32, 32]} />
          <meshPhysicalMaterial
            color="#131A2A"
            emissive="#3B82F6"
            emissiveIntensity={0.35}
            roughness={0.15}
            metalness={0.2}
            transparent={true}
            opacity={glassOpacity}
            transmission={0.6}
            ior={1.2}
          />
        </mesh>
      </group>

      {/* --- CEREBRUM RIGHT HEMISPHERE --- */}
      <group position={[0.48, 0.15, 0]}>
        {/* Wireframe Shell */}
        <mesh>
          <sphereGeometry args={[0.72, 28, 28]} />
          <meshStandardMaterial
            color="#22D3EE"
            wireframe={true}
            transparent={true}
            opacity={wireframeOpacity * 0.4}
          />
        </mesh>
        {/* Holographic Translucent Glass Body */}
        <mesh>
          <sphereGeometry args={[0.7, 32, 32]} />
          <meshPhysicalMaterial
            color="#131A2A"
            emissive="#3B82F6"
            emissiveIntensity={0.35}
            roughness={0.15}
            metalness={0.2}
            transparent={true}
            opacity={glassOpacity}
            transmission={0.6}
            ior={1.2}
          />
        </mesh>

        {/* --- LEVEL 2 ORGANIC GLOWING TUMOR (Embedded inside Right Hemisphere) --- */}
        <group ref={tumorRef} position={[0.15, 0.1, 0.1]}>
          {/* Main Core Blob */}
          <mesh>
            <sphereGeometry args={[0.26, 24, 24]} />
            <meshStandardMaterial
              color="#EF4444"
              emissive="#EF4444"
              emissiveIntensity={tumorProgress * 2.5}
              transparent={true}
              opacity={tumorProgress * 0.9}
              roughness={0.3}
            />
          </mesh>
          {/* Organic Nodule Satellite Blob 1 */}
          <mesh position={[0.12, 0.08, -0.06]}>
            <sphereGeometry args={[0.16, 20, 20]} />
            <meshStandardMaterial
              color="#EF4444"
              emissive="#EF4444"
              emissiveIntensity={tumorProgress * 2.2}
              transparent={true}
              opacity={tumorProgress * 0.85}
            />
          </mesh>
          {/* Organic Nodule Satellite Blob 2 */}
          <mesh position={[-0.08, -0.1, 0.08]}>
            <sphereGeometry args={[0.14, 20, 20]} />
            <meshStandardMaterial
              color="#EF4444"
              emissive="#EF4444"
              emissiveIntensity={tumorProgress * 2.2}
              transparent={true}
              opacity={tumorProgress * 0.85}
            />
          </mesh>
        </group>
      </group>

      {/* --- CEREBELLUM (Posterior Lower Brain) --- */}
      <group position={[0, -0.55, -0.3]}>
        <mesh>
          <sphereGeometry args={[0.48, 24, 24]} />
          <meshStandardMaterial
            color="#22D3EE"
            wireframe={true}
            transparent={true}
            opacity={wireframeOpacity * 0.3}
          />
        </mesh>
        <mesh>
          <sphereGeometry args={[0.46, 24, 24]} />
          <meshPhysicalMaterial
            color="#131A2A"
            emissive="#3B82F6"
            emissiveIntensity={0.25}
            transparent={true}
            opacity={glassOpacity * 0.9}
          />
        </mesh>
      </group>

      {/* --- BRAINSTEM (Inferior Base Structure) --- */}
      <group position={[0, -0.85, -0.1]} rotation={[0.2, 0, 0]}>
        <mesh>
          <cylinderGeometry args={[0.18, 0.14, 0.6, 16]} />
          <meshStandardMaterial
            color="#22D3EE"
            wireframe={true}
            transparent={true}
            opacity={wireframeOpacity * 0.3}
          />
        </mesh>
        <mesh>
          <cylinderGeometry args={[0.17, 0.13, 0.58, 16]} />
          <meshPhysicalMaterial
            color="#131A2A"
            emissive="#3B82F6"
            emissiveIntensity={0.2}
            transparent={true}
            opacity={glassOpacity * 0.8}
          />
        </mesh>
      </group>
    </group>
  );
}

// ----------------------------------------------------------------------
// 2. Diagnostic Target Rings (X, Y, Z Orbital HUD Plane Lines)
// ----------------------------------------------------------------------
function DiagnosticRings({ progress }: { progress: number }) {
  const ring1Ref = useRef<THREE.Group>(null);
  const ring2Ref = useRef<THREE.Group>(null);
  const ring3Ref = useRef<THREE.Group>(null);

  const ringOpacity = Math.min(0.6, Math.max(0, (progress - 0.4) / 0.4));

  useFrame((_, delta) => {
    if (ring1Ref.current) ring1Ref.current.rotation.x += delta * 0.25;
    if (ring2Ref.current) ring2Ref.current.rotation.y += delta * 0.35;
    if (ring3Ref.current) ring3Ref.current.rotation.z += delta * 0.2;
  });

  return (
    <group>
      {/* X Orbit Ring */}
      <group ref={ring1Ref}>
        <mesh>
          <torusGeometry args={[1.85, 0.008, 16, 100]} />
          <meshBasicMaterial
            color="#22D3EE"
            transparent={true}
            opacity={ringOpacity}
          />
        </mesh>
      </group>

      {/* Y Orbit Ring */}
      <group ref={ring2Ref}>
        <mesh>
          <torusGeometry args={[2.15, 0.008, 16, 100]} />
          <meshBasicMaterial
            color="#3B82F6"
            transparent={true}
            opacity={ringOpacity * 0.8}
          />
        </mesh>
      </group>

      {/* Z Orbit Ring */}
      <group ref={ring3Ref}>
        <mesh>
          <torusGeometry args={[2.45, 0.008, 16, 100]} />
          <meshBasicMaterial
            color="#22D3EE"
            transparent={true}
            opacity={ringOpacity * 0.6}
          />
        </mesh>
      </group>
    </group>
  );
}

// ----------------------------------------------------------------------
// 3. Diagnostic Swarm Particles
// ----------------------------------------------------------------------
function ParticleSwarm({ progress }: { progress: number }) {
  const pointsRef = useRef<THREE.Points>(null);

  const particleCount = 140;
  const positions = React.useMemo(() => {
    const pos = new Float32Array(particleCount * 3);
    for (let i = 0; i < particleCount; i++) {
      const radius = 2.0 + Math.random() * 1.5;
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);
      pos[i * 3] = radius * Math.sin(phi) * Math.cos(theta);
      pos[i * 3 + 1] = radius * Math.sin(phi) * Math.sin(theta);
      pos[i * 3 + 2] = radius * Math.cos(phi);
    }
    return pos;
  }, []);

  const particleOpacity = Math.min(0.75, progress * 0.8);

  useFrame((_, delta) => {
    if (pointsRef.current) {
      pointsRef.current.rotation.y += delta * 0.08;
      pointsRef.current.rotation.x += delta * 0.04;
    }
  });

  return (
    <points ref={pointsRef}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          args={[positions, 3]}
        />
      </bufferGeometry>
      <pointsMaterial
        size={0.035}
        color="#22D3EE"
        transparent={true}
        opacity={particleOpacity}
        blending={THREE.AdditiveBlending}
      />
    </points>
  );
}

// ----------------------------------------------------------------------
// 4. Main Exported BrainViewer Component
// ----------------------------------------------------------------------
export default function BrainViewer({
  prediction = "Glioma",
  confidence = 98.4,
}: BrainViewerProps) {
  const [mounted, setMounted] = useState(false);
  const [progress, setProgress] = useState(0);
  const [autoRotate, setAutoRotate] = useState(true);
  const timerRef = useRef<NodeJS.Timeout | null>(null);

  // Client hydration check
  useEffect(() => {
    setMounted(true);
  }, []);

  // 2-Second Materialization Sequence Timer (0.0 -> 1.0)
  useEffect(() => {
    if (!mounted) return;
    const startTime = Date.now();
    const duration = 2000; // 2 seconds

    const interval = setInterval(() => {
      const elapsed = Date.now() - startTime;
      const currentProgress = Math.min(1, elapsed / duration);
      setProgress(currentProgress);

      if (currentProgress >= 1) {
        clearInterval(interval);
      }
    }, 16);

    return () => clearInterval(interval);
  }, [mounted]);

  // Handle user interaction with OrbitControls (Pause auto-rotate -> Resume after 3s inactivity)
  const handleControlStart = () => {
    setAutoRotate(false);
    if (timerRef.current) clearTimeout(timerRef.current);
  };

  const handleControlEnd = () => {
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = setTimeout(() => {
      setAutoRotate(true);
    }, 3000); // 3 seconds inactivity threshold
  };

  if (!mounted) {
    return (
      <div className="w-full aspect-square rounded-3xl bg-[#131A2A] border border-[#263245] flex items-center justify-center font-mono text-xs text-[#94A3B8]">
        INITIALIZING 3D ENGINE...
      </div>
    );
  }

  return (
    <div className="w-full bg-[#131A2A] border border-[#263245] rounded-3xl p-5 shadow-2xl relative overflow-hidden flex flex-col gap-4 font-sans">
      {/* Header Bar */}
      <div className="flex items-center justify-between border-b border-[#263245] pb-3.5">
        <div className="flex items-center gap-2.5">
          <div className="p-2 rounded-xl bg-[#1A2336] border border-[#263245] text-[#3B82F6]">
            <Brain className="w-4 h-4 text-[#3B82F6]" />
          </div>
          <div>
            <h3 className="text-xs font-mono font-bold uppercase tracking-wider text-[#F8FAFC]">
              3D BRAIN VOLUME VISUALIZER
            </h3>
            <p className="text-[11px] text-[#94A3B8]">
              ResNet-50 Volumetric Reconstruction
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <span className="text-[10px] font-mono text-[#22D3EE] bg-[#1A2336] border border-[#263245] px-2.5 py-1 rounded">
            LEVEL 2 ORGANIC MODEL
          </span>
        </div>
      </div>

      {/* 3D WebGL Canvas Container */}
      <div className="relative w-full aspect-square sm:aspect-[4/3] rounded-2xl bg-[#0A0F1C] border border-[#263245] overflow-hidden group">
        {/* Materialization Sequence HUD Status Pill */}
        <div className="absolute top-3 left-3 z-10 flex items-center gap-2 bg-[#131A2A]/90 border border-[#263245] px-3 py-1.5 rounded-lg backdrop-blur-md text-[11px] font-mono text-[#F8FAFC]">
          <span className="w-2 h-2 rounded-full bg-[#22C55E] animate-pulse" />
          <span>
            {progress < 1
              ? `MATERIALIZING... ${Math.round(progress * 100)}%`
              : "3D BRAIN ACTIVE"}
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

        {/* Control Mode Badge */}
        <div className="absolute top-3 right-3 z-10 hidden sm:flex items-center gap-2 bg-[#131A2A]/90 border border-[#263245] px-2.5 py-1 rounded-lg backdrop-blur-md text-[10px] font-mono text-[#94A3B8]">
          <RotateCw className="w-3 h-3 text-[#22D3EE]" />
          <span>{autoRotate ? "AUTO ROTATING" : "MANUAL CONTROL"}</span>
        </div>

        {/* Three.js Canvas */}
        <Canvas
          camera={{ position: [0, 0, 4.5], fov: 45 }}
          className="w-full h-full cursor-grab active:cursor-grabbing"
        >
          <ambientLight intensity={0.6} />
          <directionalLight position={[10, 10, 10]} intensity={1.2} />
          <pointLight position={[-10, -10, -10]} color="#3B82F6" intensity={0.8} />
          <pointLight position={[2, 1, 2]} color="#EF4444" intensity={1.5} />

          <HolographicBrain progress={progress} />
          <DiagnosticRings progress={progress} />
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
      <div className="flex flex-wrap items-center justify-between gap-2 text-[11px] font-mono text-[#94A3B8] border-t border-[#263245] pt-3">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5">
            <RotateCw className="w-3.5 h-3.5 text-[#3B82F6]" />
            <span>Hold Left-Click + Drag: 360° Rotate</span>
          </div>
          <div className="flex items-center gap-1.5">
            <ZoomIn className="w-3.5 h-3.5 text-[#22D3EE]" />
            <span>Scroll: Zoom</span>
          </div>
        </div>

        <div className="flex items-center gap-1 text-[#22C55E]">
          <ShieldCheck className="w-3.5 h-3.5" />
          <span>AUTO-IDLE RESUMPTION ACTIVE</span>
        </div>
      </div>
    </div>
  );
}
