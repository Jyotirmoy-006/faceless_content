/**
 * One Piece 3D Studio Engine — Real-time WebGL physical character rendering.
 */

(function(window) {
  'use strict';

  const THREE = window.THREE;
  if (!THREE || !window.ThreeCharacterBuilder) {
    console.warn('Three.js or ThreeCharacterBuilder not ready.');
    return;
  }

  function isWebGLAvailable() {
    try {
      const c = document.createElement('canvas');
      return !!(window.WebGLRenderingContext && (c.getContext('webgl') || c.getContext('experimental-webgl')));
    } catch (e) {
      return false;
    }
  }

  class OnePiece3DEngine {
    constructor(container, studioInstance) {
      this.container = container;
      this.studio = studioInstance;
      this.characters = new Map();
      this.raycaster = new THREE.Raycaster();
      this.mouse = new THREE.Vector2();
      this.hoveredAgentId = null;
      this.walkTimers = new Map();
      this.isFallback = false;

      if (!isWebGLAvailable()) {
        this.triggerFallback('WebGL unsupported or disabled');
        return;
      }
      try {
        this.initScene();
        this.setupEventListeners();
      } catch (err) {
        this.triggerFallback(err.message || 'WebGL context failure');
      }
    }

    triggerFallback(reason) {
      this.isFallback = true;
      console.warn('[OnePiece3DEngine] WebGL inactive:', reason);
      if (this.renderer && this.renderer.domElement && this.renderer.domElement.parentNode) {
        this.renderer.domElement.parentNode.removeChild(this.renderer.domElement);
      }
      if (this.studio && typeof this.studio.activate2DFallback === 'function') {
        this.studio.activate2DFallback(reason);
      }
    }

    initScene() {
      // 1. Scene setup
      this.scene = new THREE.Scene();

      // 2. Orthographic Camera matching 1024x420 isometric viewport (Y=420 top, Y=0 bottom)
      this.camera = new THREE.OrthographicCamera(0, 1024, 420, 0, -1500, 1500);
      this.camera.position.set(0, 0, 500);
      this.camera.lookAt(0, 0, 0);

      // 3. WebGL Renderer with Alpha Transparency
      this.renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true, powerPreference: 'high-performance' });
      this.renderer.setSize(1024, 420);
      this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
      this.renderer.domElement.id = 'three-simulation-canvas';
      this.renderer.domElement.style.position = 'absolute';
      this.renderer.domElement.style.top = '0';
      this.renderer.domElement.style.left = '0';
      this.renderer.domElement.style.width = '100%';
      this.renderer.domElement.style.height = '100%';
      this.renderer.domElement.style.zIndex = '50';
      this.renderer.domElement.style.pointerEvents = 'auto';

      this.container.appendChild(this.renderer.domElement);

      // 4. Studio Lighting: Ambient + Directional Key Light + Soft Cool Fill
      const ambientLight = new THREE.AmbientLight(0xffffff, 0.75);
      this.scene.add(ambientLight);

      const dirLight = new THREE.DirectionalLight(0xfff7ed, 0.85);
      dirLight.position.set(200, 400, 300);
      this.scene.add(dirLight);

      const fillLight = new THREE.DirectionalLight(0x93c5fd, 0.35);
      fillLight.position.set(-200, 200, 200);
      this.scene.add(fillLight);

      // 5. Build 3D rigs for all 10 Straw Hat agents
      if (this.studio && this.studio.agents) {
        this.studio.agents.forEach(agent => {
          const rig = window.ThreeCharacterBuilder.buildCharacterRig(agent.id);
          rig.rootGroup.position.set(agent.x, 420 - agent.y, agent.y);
          // Isometric 3D tilt towards camera
          rig.rootGroup.rotation.x = 0.35;
          rig.targetYaw = 0;
          rig.currentYaw = 0;
          this.scene.add(rig.rootGroup);
          this.characters.set(agent.id, rig);
          this.walkTimers.set(agent.id, 0);
        });
      }

      // 6. Add 3D furniture occlusion cutouts (Z-Buffer sorting)
      this.addFurnitureOccluders();
    }

    addFurnitureOccluders() {
      if (!this.studio || !this.studio.FURNITURE_OVERLAYS) return;
      const loader = new THREE.TextureLoader();
      this.studio.FURNITURE_OVERLAYS.forEach(f => {
        loader.load(
          f.src,
          (texture) => {
            const geo = new THREE.PlaneGeometry(f.w, f.h);
            const mat = new THREE.MeshBasicMaterial({ map: texture, transparent: true, alphaTest: 0.1 });
            const mesh = new THREE.Mesh(geo, mat);
            mesh.position.set(f.x + f.w / 2, 420 - (f.y + f.h / 2), f.sortY);
            this.scene.add(mesh);
          },
          undefined,
          (err) => console.warn(`[OnePiece3D] Furniture cutout load skipped (${f.src}):`, err)
        );
      });
    }

    setupEventListeners() {
      const canvas = this.renderer.domElement;

      // Mouse Move for Hover Cursor
      canvas.addEventListener('mousemove', (e) => {
        const rect = canvas.getBoundingClientRect();
        this.mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
        this.mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

        this.raycaster.setFromCamera(this.mouse, this.camera);
        const intersects = this.raycaster.intersectObjects(this.scene.children, true);

        if (intersects.length > 0) {
          const hitRoot = this.findParentCharacterRoot(intersects[0].object);
          if (hitRoot) {
            canvas.style.cursor = 'pointer';
            return;
          }
        }
        canvas.style.cursor = 'default';
      });

      // Click to Inspect
      canvas.addEventListener('click', (e) => {
        const rect = canvas.getBoundingClientRect();
        this.mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
        this.mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

        this.raycaster.setFromCamera(this.mouse, this.camera);
        const intersects = this.raycaster.intersectObjects(this.scene.children, true);

        if (intersects.length > 0) {
          const hitRoot = this.findParentCharacterRoot(intersects[0].object);
          if (hitRoot && this.studio) {
            const agentId = hitRoot.name.replace('character_root_', '');
            this.studio.inspectAgent(agentId);
          }
        }
      });
    }

    findParentCharacterRoot(obj) {
      let curr = obj;
      while (curr) {
        if (curr.name && curr.name.startsWith('character_root_')) {
          return curr;
        }
        curr = curr.parent;
      }
      return null;
    }

    /**
     * Updates 3D kinematics per frame:
     * - Dual-leg stride & arm swing
     * - Physical yaw rotation matching movement angle
     * - Grounding shadow scaling
     * - Idle breathing bob / working typing
     */
    update(dt, timestamp) {
      if (this.isFallback || !this.studio || !this.studio.agents || !this.renderer) return;

      this.studio.agents.forEach(agent => {
        const rig = this.characters.get(agent.id);
        if (!rig) return;

        // Sync 3D position & Z-Depth to agent isometric coordinates (Y=420 top, Y=0 bottom)
        rig.rootGroup.position.x = agent.x;
        rig.rootGroup.position.y = 420 - agent.y;
        // Z-Depth allows physically correct occlusion in 3D
        rig.rootGroup.position.z = agent.y;

        const isWalking = (agent.path && agent.path.length > 0);
        let walkTimer = this.walkTimers.get(agent.id) || 0;

        if (isWalking) {
          walkTimer += dt * 11.5;
          this.walkTimers.set(agent.id, walkTimer);

          // 1. Dual-Leg Walking Stride
          rig.leftLegGroup.rotation.x = Math.sin(walkTimer) * 0.75;
          rig.rightLegGroup.rotation.x = -Math.sin(walkTimer) * 0.75;

          // 2. Opposite Arm Swing
          rig.leftArmGroup.rotation.x = -Math.sin(walkTimer) * 0.65;
          rig.rightArmGroup.rotation.x = Math.sin(walkTimer) * 0.65;

          // 3. Physical Vertical Bounce
          const bounce = Math.abs(Math.sin(walkTimer)) * 2.4;
          rig.bodyGroup.position.y = bounce;

          // 4. Grounding Shadow Dynamics (scales down at apex, expands on contact)
          const shadowScale = 1.0 - (bounce / 2.4) * 0.3;
          rig.shadowMesh.scale.set(shadowScale, shadowScale, shadowScale);
          rig.shadowMesh.material.opacity = 0.65 - (bounce / 2.4) * 0.3;

          // 5. 3D Yaw Rotation towards walking destination
          const nextNode = this.studio.NODES[agent.path[0]];
          if (nextNode) {
            const dx = nextNode.x - agent.x;
            const dy = nextNode.y - agent.y;
            if (Math.hypot(dx, dy) > 1.5) {
              // Calculate 3D heading angle
              rig.targetYaw = Math.atan2(dx, dy);
            }
          }

          // Smooth shortest-angle yaw interpolation
          let diff = rig.targetYaw - (rig.currentYaw || 0);
          while (diff < -Math.PI) diff += Math.PI * 2;
          while (diff > Math.PI) diff -= Math.PI * 2;
          rig.currentYaw = (rig.currentYaw || 0) + diff * 0.25;
          rig.rootGroup.rotation.y = rig.currentYaw;

          // Head slight stride bob
          rig.headGroup.rotation.x = Math.sin(walkTimer) * 0.08;

        } else {
          // Reset leg/arm stride
          rig.leftLegGroup.rotation.x += (0 - rig.leftLegGroup.rotation.x) * 0.15;
          rig.rightLegGroup.rotation.x += (0 - rig.rightLegGroup.rotation.x) * 0.15;

          // Return yaw towards front-facing
          rig.targetYaw = 0;
          let diff = rig.targetYaw - (rig.currentYaw || 0);
          while (diff < -Math.PI) diff += Math.PI * 2;
          while (diff > Math.PI) diff -= Math.PI * 2;
          rig.currentYaw = (rig.currentYaw || 0) + diff * 0.12;
          rig.rootGroup.rotation.y = rig.currentYaw;

          if (agent.state === 'EXECUTING') {
            // Working: arms raised forward typing/reviewing
            const workPhase = timestamp * 0.009 + (agent.idlePhase || 0);
            rig.leftArmGroup.rotation.x = -1.1 + Math.sin(workPhase) * 0.25;
            rig.rightArmGroup.rotation.x = -1.1 - Math.sin(workPhase) * 0.25;
            rig.headGroup.rotation.x = 0.2 + Math.sin(workPhase) * 0.1;
            rig.bodyGroup.position.y = Math.sin(workPhase) * 0.8;
            rig.shadowMesh.scale.set(1.0, 1.0, 1.0);
            rig.shadowMesh.material.opacity = 0.65;
          } else {
            // Idle: Gentle breathing rhythm
            const breathPhase = timestamp * 0.0025 + (agent.idlePhase || 0);
            const breath = Math.sin(breathPhase);
            rig.bodyGroup.position.y = breath * 0.6;
            rig.leftArmGroup.rotation.x += (0 - rig.leftArmGroup.rotation.x) * 0.1;
            rig.rightArmGroup.rotation.x += (0 - rig.rightArmGroup.rotation.x) * 0.1;
            rig.headGroup.rotation.x = breath * 0.04;
            const shadowS = 1.0 - breath * 0.05;
            rig.shadowMesh.scale.set(shadowS, shadowS, shadowS);
            rig.shadowMesh.material.opacity = 0.55 + breath * 0.08;
          }
        }
      });

      // Render 3D Scene
      this.renderer.render(this.scene, this.camera);
    }
  }

  window.OnePiece3DEngine = OnePiece3DEngine;

})(window);
