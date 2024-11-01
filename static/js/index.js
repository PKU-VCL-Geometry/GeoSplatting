let tab = -1;

function tab_click1() {
  if (tab == 1) return;
  if (tab != -1) {
    document.getElementById('tab-' + tab.toString()).classList.toggle('is-active');
  }
  tab = 1;
  document.getElementById('tab-1').classList.toggle('is-active');
  document.getElementById('player-1-1').src = './static/videos/spot.bridge.mp4';
  document.getElementById('player-1-2').src = './static/videos/spot.city.mp4';
  document.getElementById('player-1-3').src = './static/videos/spot.fireplace.mp4';
  document.getElementById('player-1-4').src = './static/videos/spot.forest.mp4';
  document.getElementById('player-1-5').src = './static/videos/spot.night.mp4';
  document.getElementById('player-2-1').src = './static/videos/spot.fix.bridge.mp4';
  document.getElementById('player-2-2').src = './static/videos/spot.fix.city.mp4';
  document.getElementById('player-2-3').src = './static/videos/spot.fix.fireplace.mp4';
  document.getElementById('player-2-4').src = './static/videos/spot.fix.forest.mp4';
  document.getElementById('player-2-5').src = './static/videos/spot.fix.night.mp4';
  document.getElementById('player-1-1').load();
  document.getElementById('player-1-2').load();
  document.getElementById('player-1-3').load();
  document.getElementById('player-1-4').load();
  document.getElementById('player-1-5').load();
  document.getElementById('player-2-1').load();
  document.getElementById('player-2-2').load();
  document.getElementById('player-2-3').load();
  document.getElementById('player-2-4').load();
  document.getElementById('player-2-5').load();
}

function tab_click2() {
  if (tab == 2) return;
  if (tab != -1) {
    document.getElementById('tab-' + tab.toString()).classList.toggle('is-active');
  }
  tab = 2;
  document.getElementById('tab-2').classList.toggle('is-active');
  document.getElementById('player-1-1').src = './static/videos/materials.bridge.mp4';
  document.getElementById('player-1-2').src = './static/videos/materials.city.mp4';
  document.getElementById('player-1-3').src = './static/videos/materials.fireplace.mp4';
  document.getElementById('player-1-4').src = './static/videos/materials.forest.mp4';
  document.getElementById('player-1-5').src = './static/videos/materials.night.mp4';
  document.getElementById('player-2-1').src = './static/videos/materials.fix.bridge.mp4';
  document.getElementById('player-2-2').src = './static/videos/materials.fix.city.mp4';
  document.getElementById('player-2-3').src = './static/videos/materials.fix.fireplace.mp4';
  document.getElementById('player-2-4').src = './static/videos/materials.fix.forest.mp4';
  document.getElementById('player-2-5').src = './static/videos/materials.fix.night.mp4';
  document.getElementById('player-1-1').load();
  document.getElementById('player-1-2').load();
  document.getElementById('player-1-3').load();
  document.getElementById('player-1-4').load();
  document.getElementById('player-1-5').load();
  document.getElementById('player-2-1').load();
  document.getElementById('player-2-2').load();
  document.getElementById('player-2-3').load();
  document.getElementById('player-2-4').load();
  document.getElementById('player-2-5').load();
}

function tab_click3() {
  if (tab == 3) return;
  if (tab != -1) {
    document.getElementById('tab-' + tab.toString()).classList.toggle('is-active');
  }
  tab = 3;
  document.getElementById('tab-3').classList.toggle('is-active');
  document.getElementById('player-1-1').src = './static/videos/arm.bridge.mp4';
  document.getElementById('player-1-2').src = './static/videos/arm.city.mp4';
  document.getElementById('player-1-3').src = './static/videos/arm.fireplace.mp4';
  document.getElementById('player-1-4').src = './static/videos/arm.forest.mp4';
  document.getElementById('player-1-5').src = './static/videos/arm.night.mp4';
  document.getElementById('player-2-1').src = './static/videos/arm.fix.bridge.mp4';
  document.getElementById('player-2-2').src = './static/videos/arm.fix.city.mp4';
  document.getElementById('player-2-3').src = './static/videos/arm.fix.fireplace.mp4';
  document.getElementById('player-2-4').src = './static/videos/arm.fix.forest.mp4';
  document.getElementById('player-2-5').src = './static/videos/arm.fix.night.mp4';
  document.getElementById('player-1-1').load();
  document.getElementById('player-1-2').load();
  document.getElementById('player-1-3').load();
  document.getElementById('player-1-4').load();
  document.getElementById('player-1-5').load();
  document.getElementById('player-2-1').load();
  document.getElementById('player-2-2').load();
  document.getElementById('player-2-3').load();
  document.getElementById('player-2-4').load();
  document.getElementById('player-2-5').load();
}
