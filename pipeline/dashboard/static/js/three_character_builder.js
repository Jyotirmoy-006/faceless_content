/**
 * Three.js Procedural 3D Character Mesh Builder for Straw Hat Studio.
 *
 * Constructs articulated physical 3D character rigs with defined body parts:
 * - Pelvis / Root
 * - Torso (with iconic attire)
 * - Head Group (head mesh, expressive face, styled hair, signature hats)
 * - Left & Right Shoulder / Arm Groups (with tools & weapons)
 * - Left & Right Hip / Leg Groups (with shoes/sandals)
 * - Grounding Shadow Mesh
 */

(function(window) {
  'use strict';

  const THREE = window.THREE;
  if (!THREE) {
    console.error('Three.js not found for CharacterBuilder.');
    return;
  }

  // Common Materials & Color Palette
  const MATS = {
    skin: new THREE.MeshLambertMaterial({ color: 0xffdfc4 }),
    skinZoro: new THREE.MeshLambertMaterial({ color: 0xf5d3b3 }),
    skinRobin: new THREE.MeshLambertMaterial({ color: 0xfde2cd }),
    skinJinbe: new THREE.MeshLambertMaterial({ color: 0x4aa3df }),
    skinBrook: new THREE.MeshLambertMaterial({ color: 0xeeeeee }),
    red: new THREE.MeshLambertMaterial({ color: 0xdc2626 }),
    blue: new THREE.MeshLambertMaterial({ color: 0x2563eb }),
    cyan: new THREE.MeshLambertMaterial({ color: 0x06b6d4 }),
    straw: new THREE.MeshLambertMaterial({ color: 0xeab308 }),
    strawRibbon: new THREE.MeshLambertMaterial({ color: 0xb91c1c }),
    greenZoro: new THREE.MeshLambertMaterial({ color: 0x15803d }),
    greenHaramaki: new THREE.MeshLambertMaterial({ color: 0x4ade80 }),
    black: new THREE.MeshLambertMaterial({ color: 0x18181b }),
    gold: new THREE.MeshLambertMaterial({ color: 0xf59e0b }),
    orangeNami: new THREE.MeshLambertMaterial({ color: 0xf97316 }),
    white: new THREE.MeshLambertMaterial({ color: 0xf4f4f5 }),
    brown: new THREE.MeshLambertMaterial({ color: 0x78350f }),
    pinkChopper: new THREE.MeshLambertMaterial({ color: 0xec4899 }),
    blueChopperNose: new THREE.MeshLambertMaterial({ color: 0x38bdf8 }),
    purpleRobin: new THREE.MeshLambertMaterial({ color: 0x7e22ce }),
    darkTeal: new THREE.MeshLambertMaterial({ color: 0x0f766e }),
    shadow: new THREE.MeshBasicMaterial({ color: 0x09090b, transparent: true, opacity: 0.65 }),
  };

  /** Helper to create a box mesh */
  function box(w, h, d, mat, x = 0, y = 0, z = 0) {
    const geo = new THREE.BoxGeometry(w, h, d);
    const m = new THREE.Mesh(geo, mat);
    m.position.set(x, y, z);
    return m;
  }

  /** Helper to create a cylinder mesh */
  function cyl(rt, rb, h, segs, mat, x = 0, y = 0, z = 0) {
    const geo = new THREE.CylinderGeometry(rt, rb, h, segs);
    const m = new THREE.Mesh(geo, mat);
    m.position.set(x, y, z);
    return m;
  }

  /**
   * Builds an articulated 3D character model for a given agentId.
   * Returns an object containing the root group and references to animated limbs.
   */
  function buildCharacterRig(agentId) {
    const root = new THREE.Group();
    root.name = `character_root_${agentId}`;

    // 1. Soft Grounding Shadow
    const shadowGeo = new THREE.PlaneGeometry(16, 8);
    const shadow = new THREE.Mesh(shadowGeo, MATS.shadow);
    shadow.rotation.x = -Math.PI / 2;
    shadow.position.set(0, 0.2, 0);
    root.add(shadow);

    // 2. Character Body Container (handles vertical bob & squash-stretch)
    const body = new THREE.Group();
    body.position.y = 0;
    root.add(body);

    // 3. Legs (Pivoted at hips)
    const leftLegGroup = new THREE.Group();
    leftLegGroup.position.set(-3.5, 9, 0);
    const rightLegGroup = new THREE.Group();
    rightLegGroup.position.set(3.5, 9, 0);

    const legMat = (agentId === 'zoro' || agentId === 'sanji' || agentId === 'brook') ? MATS.black
      : (agentId === 'luffy') ? MATS.blue
      : (agentId === 'jinbe') ? MATS.skinJinbe
      : MATS.brown;

    const leftLegMesh = box(3.2, 9, 3.2, legMat, 0, -4.5, 0);
    const rightLegMesh = box(3.2, 9, 3.2, legMat, 0, -4.5, 0);
    leftLegGroup.add(leftLegMesh);
    rightLegGroup.add(rightLegMesh);
    body.add(leftLegGroup);
    body.add(rightLegGroup);

    // 4. Torso
    let torsoMat = MATS.red;
    let torsoW = 11, torsoH = 12, torsoD = 6;
    if (agentId === 'zoro') torsoMat = MATS.greenHaramaki;
    else if (agentId === 'nami') torsoMat = MATS.blue;
    else if (agentId === 'usopp') torsoMat = MATS.brown;
    else if (agentId === 'sanji') torsoMat = MATS.black;
    else if (agentId === 'chopper') { torsoMat = MATS.brown; torsoW = 8; torsoH = 8; }
    else if (agentId === 'robin') torsoMat = MATS.purpleRobin;
    else if (agentId === 'franky') { torsoMat = MATS.red; torsoW = 16; torsoH = 13; torsoD = 8; }
    else if (agentId === 'brook') { torsoMat = MATS.black; torsoW = 9; torsoH = 14; }
    else if (agentId === 'jinbe') { torsoMat = MATS.gold; torsoW = 18; torsoH = 14; torsoD = 9; }

    const torso = box(torsoW, torsoH, torsoD, torsoMat, 0, 9 + torsoH / 2, 0);
    body.add(torso);

    // 5. Arms (Pivoted at shoulders)
    const shoulderY = 9 + torsoH - 1.5;
    const armW = (agentId === 'franky') ? 5.5 : 3.0;
    const armH = (agentId === 'brook') ? 11 : 8.5;
    const leftArmGroup = new THREE.Group();
    leftArmGroup.position.set(-(torsoW / 2 + armW / 2), shoulderY, 0);
    const rightArmGroup = new THREE.Group();
    rightArmGroup.position.set(torsoW / 2 + armW / 2, shoulderY, 0);

    const armMat = (agentId === 'franky') ? MATS.cyan
      : (agentId === 'sanji' || agentId === 'brook') ? MATS.black
      : (agentId === 'jinbe') ? MATS.skinJinbe
      : (agentId === 'luffy' || agentId === 'nami') ? MATS.skin
      : torsoMat;

    const leftArmMesh = box(armW, armH, armW, armMat, 0, -armH / 2, 0);
    const rightArmMesh = box(armW, armH, armW, armMat, 0, -armH / 2, 0);
    leftArmGroup.add(leftArmMesh);
    rightArmGroup.add(rightArmMesh);
    body.add(leftArmGroup);
    body.add(rightArmGroup);

    // Signature Props on Arms
    if (agentId === 'zoro') {
      const sword = box(1.2, 14, 1.2, MATS.white, 0, -3, 3);
      sword.rotation.z = 0.4;
      leftArmGroup.add(sword);
    } else if (agentId === 'usopp') {
      const slingshot = cyl(1.5, 0.8, 6, 6, MATS.brown, 0, -4, 2);
      rightArmGroup.add(slingshot);
    } else if (agentId === 'brook') {
      const cane = cyl(0.6, 0.6, 14, 6, MATS.gold, 0, -5, 2);
      rightArmGroup.add(cane);
    }

    // 6. Head Group
    const headGroup = new THREE.Group();
    const headY = 9 + torsoH;
    headGroup.position.set(0, headY, 0);

    const skinMat = (agentId === 'jinbe') ? MATS.skinJinbe
      : (agentId === 'brook') ? MATS.skinBrook
      : (agentId === 'zoro') ? MATS.skinZoro
      : (agentId === 'robin') ? MATS.skinRobin
      : MATS.skin;

    const headW = (agentId === 'jinbe') ? 12 : 9;
    const headH = 8.5;
    const headD = 8.5;
    const headMesh = box(headW, headH, headD, skinMat, 0, headH / 2, 0);
    headGroup.add(headMesh);

    // Facial features: Anime Eyes
    const eyeMat = MATS.black;
    const leftEye = box(1.4, 1.4, 0.6, eyeMat, -2.2, headH / 2, headD / 2 + 0.1);
    const rightEye = box(1.4, 1.4, 0.6, eyeMat, 2.2, headH / 2, headD / 2 + 0.1);
    headGroup.add(leftEye);
    headGroup.add(rightEye);

    // 7. Signature Hats & Hair
    if (agentId === 'luffy') {
      // Luffy's Straw Hat
      const brim = cyl(9.5, 9.5, 0.8, 16, MATS.straw, 0, headH + 0.2, 0);
      const crown = cyl(5.5, 6.0, 3.5, 16, MATS.straw, 0, headH + 2.0, 0);
      const ribbon = cyl(5.7, 5.7, 1.0, 16, MATS.strawRibbon, 0, headH + 1.0, 0);
      headGroup.add(brim);
      headGroup.add(crown);
      headGroup.add(ribbon);
    } else if (agentId === 'zoro') {
      // Green Marimo Hair
      const hair = box(headW + 0.4, 3.0, headD + 0.4, MATS.greenZoro, 0, headH + 0.8, 0);
      headGroup.add(hair);
    } else if (agentId === 'nami') {
      // Long Orange Hair
      const hairTop = box(headW + 0.6, 3.5, headD + 0.6, MATS.orangeNami, 0, headH + 0.8, 0);
      const hairBack = box(headW + 0.6, 9.0, 2.5, MATS.orangeNami, 0, headH / 2, -(headD / 2 + 1));
      headGroup.add(hairTop);
      headGroup.add(hairBack);
    } else if (agentId === 'usopp') {
      // Long Nose + Sniper Goggles
      const nose = cyl(0.8, 0.6, 5.0, 8, skinMat, 0, headH / 2 - 1, headD / 2 + 2.5);
      nose.rotation.x = Math.PI / 2;
      const goggles = box(headW + 0.4, 2.0, 2.0, MATS.brown, 0, headH + 0.5, headD / 2 - 1);
      headGroup.add(nose);
      headGroup.add(goggles);
    } else if (agentId === 'sanji') {
      // Blonde Hair with fringe
      const hair = box(headW + 0.6, 3.5, headD + 0.6, MATS.gold, 0, headH + 0.8, 0);
      const fringe = box(3.5, 4.0, 1.0, MATS.gold, -2.5, headH / 2 + 1.5, headD / 2 + 0.4);
      headGroup.add(hair);
      headGroup.add(fringe);
    } else if (agentId === 'chopper') {
      // Pink Top Hat with white X
      const hatBrim = cyl(7.5, 7.5, 0.8, 14, MATS.pinkChopper, 0, headH + 0.2, 0);
      const hatTop = cyl(5.0, 5.5, 6.5, 14, MATS.pinkChopper, 0, headH + 3.5, 0);
      const crossH = box(3.2, 0.8, 0.4, MATS.white, 0, headH + 3.5, 5.2);
      const crossV = box(0.8, 3.2, 0.4, MATS.white, 0, headH + 3.5, 5.2);
      const blueNose = box(1.6, 1.2, 1.2, MATS.blueChopperNose, 0, headH / 2 - 1.2, headD / 2 + 0.4);
      headGroup.add(hatBrim);
      headGroup.add(hatTop);
      headGroup.add(crossH);
      headGroup.add(crossV);
      headGroup.add(blueNose);
    } else if (agentId === 'robin') {
      // Sleek Black Hair + Sunglasses
      const hair = box(headW + 0.6, 4.0, headD + 0.8, MATS.black, 0, headH + 1.0, 0);
      const shades = box(7.0, 1.8, 1.0, MATS.gold, 0, headH + 1.8, headD / 2 - 0.5);
      headGroup.add(hair);
      headGroup.add(shades);
    } else if (agentId === 'franky') {
      // Giant Blue Pompadour
      const pomp = cyl(3.5, 3.0, 8.0, 8, MATS.cyan, 0, headH + 4.5, 1.5);
      pomp.rotation.x = -0.4;
      headGroup.add(pomp);
    } else if (agentId === 'brook') {
      // Giant Spherical Afro
      const afro = new THREE.Mesh(new THREE.SphereGeometry(7.5, 10, 10), MATS.black);
      afro.position.set(0, headH + 4.5, 0);
      headGroup.add(afro);
    } else if (agentId === 'jinbe') {
      // Topknot and Lightning Sideburns
      const topknot = cyl(1.8, 2.2, 3.5, 8, MATS.black, 0, headH + 1.8, -2);
      headGroup.add(topknot);
    }

    body.add(headGroup);

    // Character Proportions Matching One Piece Canon
    const s = (agentId === 'chopper') ? 0.85
      : (agentId === 'franky' || agentId === 'jinbe') ? 1.25
      : 1.15;
    root.scale.set(s, s, s);

    return {
      id: agentId,
      rootGroup: root,
      bodyGroup: body,
      shadowMesh: shadow,
      headGroup: headGroup,
      leftArmGroup: leftArmGroup,
      rightArmGroup: rightArmGroup,
      leftLegGroup: leftLegGroup,
      rightLegGroup: rightLegGroup,
    };
  }

  window.ThreeCharacterBuilder = {
    buildCharacterRig,
  };

})(window);
