/* =========================================================================
   CESSNA 172P — Detailed Procedural 3D Model (Three.js)
   -------------------------------------------------------------------------
   Builds a recognisable low-poly Skyhawk from primitives with:
     - Tapered fuselage (nose cone + cabin + tail cone)
     - High wing with dihedral and blue livery stripe
     - Engine cowling with emissive EGT-driven tint
     - 2-blade propeller with motion-blur disc
     - Fixed tricycle landing gear
     - Windshield + cabin windows
     - Vertical stabiliser + rudder + horizontal stabiliser
     - Wing struts, antenna mast, nav light bumps
     - State-driven animations: prop RPM, EGT glow, anomaly flash

   Usage (loaded as <script>, Cessna172 available on window):
     const plane = new Cessna172(scene);
     plane.update({ rpm: 2300, egt: 1290, is_anomaly: false });
   ========================================================================= */
"use strict";

class Cessna172 {
  constructor(scene) {
    this.scene = scene;
    this.propGroup = null;
    this.blurDisc = null;
    this.engineMats = [];
    this.navLights = [];
    this.baseEmissive = 0.55;
    this._build();
  }

  /* ---- helpers ---------------------------------------------------------- */
  _mat(color, opts = {}) {
    return new THREE.MeshStandardMaterial({
      color, roughness: opts.r ?? 0.55, metalness: opts.m ?? 0.08,
      emissive: opts.emissive ?? 0x000000,
      emissiveIntensity: opts.ei ?? 0,
      transparent: opts.transparent ?? false,
      opacity: opts.opacity ?? 1,
    });
  }

  _mesh(geo, mat, x, y, z, rx, ry, rz) {
    const m = new THREE.Mesh(geo, mat);
    m.position.set(x, y, z);
    m.rotation.set(rx || 0, ry || 0, rz || 0);
    m.castShadow = true;
    m.receiveShadow = true;
    this.scene.add(m);
    return m;
  }

  /* ---- build the full aircraft ----------------------------------------- */
  _build() {
    const bodyWhite = this._mat(0xe8ecf1);
    const stripeBlue = this._mat(0x2b6cb0, { m: 0.15 });
    const darkMat = this._mat(0x1c2129, { r: 0.85 });
    const chrome = this._mat(0x99aabb, { m: 0.6, r: 0.3 });
    const glass = this._mat(0x88bbdd, { r: 0.1, m: 0.0, transparent: true, opacity: 0.35 });

    // ---- FUSELAGE (nose → +Z) ----
    // Nose cone
    this._mesh(new THREE.ConeGeometry(0.42, 0.8, 14), bodyWhite,
               0, 0.95, 3.1, -Math.PI / 2);
    // Front fuselage (cabin forward)
    this._mesh(new THREE.CylinderGeometry(0.58, 0.62, 1.4, 14), bodyWhite,
               0, 0.95, 2.15, Math.PI / 2);
    // Cabin (wider section)
    this._mesh(new THREE.CylinderGeometry(0.62, 0.58, 1.6, 14), bodyWhite,
               0, 0.95, 0.75, Math.PI / 2);
    // Rear fuselage
    this._mesh(new THREE.CylinderGeometry(0.58, 0.30, 2.0, 12), bodyWhite,
               0, 0.95, -0.85, Math.PI / 2);
    // Tail cone (thin)
    this._mesh(new THREE.CylinderGeometry(0.30, 0.10, 1.2, 10), bodyWhite,
               0, 0.95, -2.55, Math.PI / 2);

    // Blue stripe along fuselage side
    [-1, 1].forEach(s => {
      this._mesh(new THREE.BoxGeometry(0.02, 0.18, 5.8), stripeBlue,
                 s * 0.58, 0.95, 0.2);
    });

    // ---- ENGINE COWLING (emissive driven by EGT) ----
    const cowlMat = this._mat(0xdfe6ee, { r: 0.4, m: 0.25,
      emissive: 0x1e66ff, ei: this.baseEmissive });
    this._mesh(new THREE.CylinderGeometry(0.44, 0.58, 1.05, 14), cowlMat,
               0, 0.95, 2.85, Math.PI / 2);
    this.engineMats.push(cowlMat);

    // Air intake (dark opening below cowl)
    this._mesh(new THREE.BoxGeometry(0.36, 0.12, 0.08), darkMat,
               0, 0.62, 3.35);

    // Exhaust pipes (chrome)
    [-0.18, 0.18].forEach(x => {
      this._mesh(new THREE.CylinderGeometry(0.025, 0.025, 0.3, 6), chrome,
                 x, 0.78, 3.2, Math.PI / 2);
    });

    // ---- HIGH WING ----
    // Main wing panels (span along X)
    this._mesh(new THREE.BoxGeometry(10.8, 0.10, 1.7), bodyWhite,
               0, 1.78, 0.3);
    // Wing tips (slight dihedral)
    [-1, 1].forEach(s => {
      this._mesh(new THREE.BoxGeometry(0.6, 0.08, 1.4), bodyWhite,
                 s * 5.4, 1.82, 0.3, 0, 0, s * 0.04);
    });
    // Blue leading-edge stripe
    this._mesh(new THREE.BoxGeometry(10.6, 0.04, 0.24), stripeBlue,
               0, 1.84, 1.1);
    // Ailerons (trailing edge)
    [-1, 1].forEach(s => {
      this._mesh(new THREE.BoxGeometry(1.8, 0.04, 0.35), stripeBlue,
                 s * 4.2, 1.72, -0.5);
    });
    // Flaps (inner trailing edge)
    [-1, 1].forEach(s => {
      this._mesh(new THREE.BoxGeometry(1.2, 0.04, 0.35), bodyWhite,
                 s * 1.5, 1.72, -0.5);
    });

    // ---- WING STRUTS ----
    [-1, 1].forEach(s => {
      this._mesh(new THREE.CylinderGeometry(0.035, 0.035, 2.2, 6), stripeBlue,
                 s * 2.0, 0.95, 0.75, 0, 0, s * -0.40);
    });

    // ---- TAIL ----
    // Vertical stabiliser
    this._mesh(new THREE.BoxGeometry(0.06, 1.3, 0.9), stripeBlue,
               0, 1.75, -2.55);
    // Rudder
    this._mesh(new THREE.BoxGeometry(0.04, 1.1, 0.35), stripeBlue,
               0, 1.65, -3.05);
    // Horizontal stabiliser
    this._mesh(new THREE.BoxGeometry(3.0, 0.07, 0.75), bodyWhite,
               0, 1.15, -2.5);
    // Elevator
    this._mesh(new THREE.BoxGeometry(2.8, 0.04, 0.25), stripeBlue,
               0, 1.15, -2.9);

    // ---- LANDING GEAR ----
    // Main gear legs
    [-1, 1].forEach(s => {
      this._mesh(new THREE.CylinderGeometry(0.035, 0.035, 0.75, 6), chrome,
                 s * 0.52, 0.44, 0.9, 0, 0, s * 0.55);
      // Main wheels
      this._mesh(new THREE.TorusGeometry(0.14, 0.05, 8, 14), darkMat,
                 s * 0.82, 0.15, 0.9, Math.PI / 2, 0, 0);
      // Wheel pants (fairings)
      this._mesh(new THREE.BoxGeometry(0.12, 0.18, 0.32), bodyWhite,
                 s * 0.82, 0.18, 0.9);
    });
    // Nose gear
    this._mesh(new THREE.CylinderGeometry(0.025, 0.025, 0.55, 6), chrome,
               0, 0.44, 2.6, 0, 0, 0);
    this._mesh(new THREE.TorusGeometry(0.10, 0.04, 8, 12), darkMat,
               0, 0.15, 2.6, Math.PI / 2);
    // Nose wheel pant
    this._mesh(new THREE.BoxGeometry(0.10, 0.14, 0.26), bodyWhite,
               0, 0.18, 2.6);

    // ---- PROPELLER ----
    this.propGroup = new THREE.Group();
    this.propGroup.position.set(0, 0.95, 3.52);
    this.scene.add(this.propGroup);
    const dark = this._mat(0x222a35, { r: 0.6 });
    const bladeGeo = new THREE.BoxGeometry(0.14, 2.5, 0.04);
    const b1 = new THREE.Mesh(bladeGeo, dark);
    const b2 = b1.clone();
    b2.rotation.z = Math.PI / 2;
    this.propGroup.add(b1, b2);
    // Spinner (nose cone of prop)
    this._mesh(new THREE.ConeGeometry(0.08, 0.2, 8), chrome,
               0, 0.95, 3.62, -Math.PI / 2);
    // Blur disc
    this.blurDisc = new THREE.Mesh(
      new THREE.CircleGeometry(1.3, 40),
      new THREE.MeshBasicMaterial({ color: 0x93a7bd, transparent: true, opacity: 0 })
    );
    this.blurDisc.position.z = 0.02;
    this.propGroup.add(this.blurDisc);

    // ---- WINDOWS ----
    // Windshield (two panels)
    [-0.22, 0.22].forEach(x => {
      this._mesh(new THREE.PlaneGeometry(0.36, 0.38), glass,
                 x, 1.22, 2.05, -0.35);
    });
    // Side windows
    [-1, 1].forEach(s => {
      this._mesh(new THREE.PlaneGeometry(0.55, 0.30), glass,
                 s * 0.58, 1.18, 1.2, 0, s * Math.PI / 2, 0);
      // Rear side windows
      this._mesh(new THREE.PlaneGeometry(0.40, 0.25), glass,
                 s * 0.55, 1.16, 0.3, 0, s * Math.PI / 2, 0);
    });

    // ---- NAVIGATION LIGHTS ----
    const navRed = this._mat(0xff2222, { emissive: 0xff2222, ei: 0.8 });
    const navGreen = this._mat(0x22ff22, { emissive: 0x22ff22, ei: 0.8 });
    const navWhite = this._mat(0xffffff, { emissive: 0xffffff, ei: 0.6 });
    const navGeo = new THREE.SphereGeometry(0.04, 8, 6);
    const redLight = new THREE.Mesh(navGeo, navRed);
    redLight.position.set(-5.4, 1.84, 0.3);
    this.scene.add(redLight);
    const greenLight = new THREE.Mesh(navGeo, navGreen);
    greenLight.position.set(5.4, 1.84, 0.3);
    this.scene.add(greenLight);
    const whiteLight = new THREE.Mesh(navGeo, navWhite);
    whiteLight.position.set(0, 1.20, -3.1);
    this.scene.add(whiteLight);
    this.navLights = [redLight, greenLight, whiteLight];

    // ---- ANTENNA MAST ----
    this._mesh(new THREE.CylinderGeometry(0.012, 0.012, 0.35, 4), darkMat,
               0, 2.05, -0.3);
    // Com antenna (blade on top)
    this._mesh(new THREE.BoxGeometry(0.02, 0.04, 0.25), darkMat,
               0, 2.20, -0.3);

    // ---- BELLY STRIPE ----
    this._mesh(new THREE.BoxGeometry(0.02, 0.02, 5.0), stripeBlue,
               0, 0.50, 0.2);

    // ---- SMOKE PARTICLES ----
    const smokeGeo = new THREE.BufferGeometry();
    const particleCount = 150;
    const positions = new Float32Array(particleCount * 3);
    const life = new Float32Array(particleCount);
    for (let i = 0; i < particleCount; i++) {
      positions[i * 3] = 0;
      positions[i * 3 + 1] = 0;
      positions[i * 3 + 2] = 0;
      life[i] = 0; // 0 means inactive
    }
    smokeGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    smokeGeo.setAttribute('life', new THREE.BufferAttribute(life, 1));
    const smokeMat = new THREE.PointsMaterial({
      color: 0x111111,
      size: 0.6,
      transparent: true,
      opacity: 0.4,
      depthWrite: false,
    });
    this.smokeParticles = new THREE.Points(smokeGeo, smokeMat);
    this.scene.add(this.smokeParticles);
  }

  /* ---- per-frame update (call from render loop) ------------------------- */
  update(st) {
    const rpm = st.sensors ? (st.sensors.rpm || 0) : 0;
    const egt = st.sensors ? (st.sensors.egt_degf || 0) : 0;
    const anom = !!st.is_anomaly;

    // Propeller spin
    if (this.propGroup) {
      const omega = (rpm / 60) * 2 * Math.PI * 0.18;
      this.propGroup.rotation.z += omega * (1 / 60); // ~60fps assumption
      if (this.blurDisc) {
        this.blurDisc.material.opacity = clamp3d((rpm - 700) / 900, 0, 1) * 0.28;
        this.blurDisc.visible = rpm > 700;
      }
    }

    // EGT-driven emissive: blue (cool) → red (hot)
    const egtn = clamp3d((egt - 400) / 1400, 0, 1);
    const cool = new THREE.Color(0x1e66ff);
    const hot = new THREE.Color(0xff2a1a);
    const target = anom ? hot : cool.clone().lerp(hot, egtn);
    const t = performance.now() * 0.001;
    this.engineMats.forEach(m => {
      m.emissive.lerp(target, 0.06);
      m.emissiveIntensity = anom
        ? this.baseEmissive + 0.45 * (Math.sin(t * 12) * 0.5 + 0.5)
        : this.baseEmissive + 0.35 * egtn;
    });

    // Nav light blink (subtle)
    if (this.navLights.length) {
      const blink = Math.sin(t * 2) > 0.3 ? 1.0 : 0.4;
      this.navLights[2].material.emissiveIntensity = 0.6 * blink;
    }

    // Smoke particles
    if (this.smokeParticles) {
      const oil_psi = st.sensors ? (st.sensors.oil_psi || 60) : 60;
      const shouldEmit = oil_psi < 45 || egt > 1700 || anom;
      
      const positions = this.smokeParticles.geometry.attributes.position.array;
      const life = this.smokeParticles.geometry.attributes.life.array;
      
      for (let i = 0; i < life.length; i++) {
        if (life[i] <= 0 && shouldEmit && Math.random() < 0.2) {
          // Spawn new particle at engine cowling
          positions[i * 3] = (Math.random() - 0.5) * 0.5;
          positions[i * 3 + 1] = 0.95 + (Math.random() - 0.5) * 0.5;
          positions[i * 3 + 2] = 2.85 + (Math.random() - 0.5) * 0.5;
          life[i] = 1.0; // Life starts at 1.0
        } else if (life[i] > 0) {
          // Update existing particle
          life[i] -= 0.015; // Decay life
          // Move backward and slightly up and spread
          positions[i * 3] += (Math.random() - 0.5) * 0.05; // X spread
          positions[i * 3 + 1] += 0.02; // Y move up
          positions[i * 3 + 2] -= 0.15; // Z move backward
        }
      }
      this.smokeParticles.geometry.attributes.position.needsUpdate = true;
      this.smokeParticles.geometry.attributes.life.needsUpdate = true;
    }
  }
}

window.Cessna172 = Cessna172;

function clamp3d(v, lo, hi) { return Math.min(hi, Math.max(lo, v)); }
