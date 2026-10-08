const letterCanvas = document.querySelector(".letter-rain");

if (letterCanvas) {
  const context = letterCanvas.getContext("2d", { alpha: true });
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  const narrowScreen = window.matchMedia("(max-width: 650px)");
  const alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<>[]{}+-";
  const glyph = () => alphabet[Math.floor(Math.random() * alphabet.length)];
  const random = (min, max) => min + Math.random() * (max - min);
  let width = 0;
  let height = 0;
  let streams = [];
  let frame = 0;
  let lastTime = 0;

  const makeStream = (x, startY) => {
    const length = Math.floor(random(8, 20));
    return {
      x: x + random(-6, 6),
      y: startY,
      speed: random(13, 29),
      step: random(13, 17),
      characters: Array.from({ length }, glyph),
      strength: random(.68, 1)
    };
  };

  const resize = () => {
    const bounds = letterCanvas.getBoundingClientRect();
    width = bounds.width;
    height = bounds.height;
    if (!width || !height) return;
    const scale = Math.min(window.devicePixelRatio || 1, 2);
    letterCanvas.width = Math.round(width * scale);
    letterCanvas.height = Math.round(height * scale);
    context.setTransform(scale, 0, 0, scale, 0, 0);
    streams = [];
    for (let x = 15; x < width; x += 27) {
      streams.push(makeStream(x, random(0, height)));
      streams.push(makeStream(x + 11, random(-height, height)));
    }
    render(0);
  };

  const render = (delta) => {
    context.clearRect(0, 0, width, height);
    context.font = '10px "SFMono-Regular", Consolas, monospace';
    context.textAlign = "center";
    for (const stream of streams) {
      stream.y += stream.speed * delta;
      if (stream.y - stream.characters.length * stream.step > height) {
        stream.y = random(-height * .35, -20);
      }
      stream.characters.forEach((character, index) => {
        const y = stream.y - index * stream.step;
        if (y < 0 || y > height) return;
        if (delta && Math.random() < .004) stream.characters[index] = glyph();
        const edgeFade = Math.min(1, y / 100, (height - y) / 100);
        const tailFade = 1 - index / stream.characters.length;
        const alpha = (.08 + .24 * tailFade) * edgeFade * stream.strength;
        context.fillStyle = `rgba(42, 94, 59, ${alpha})`;
        context.fillText(character, stream.x, y);
      });
    }
  };

  const animate = (time) => {
    frame = window.requestAnimationFrame(animate);
    if (time - lastTime < 32) return;
    const delta = lastTime ? Math.min((time - lastTime) / 1000, .1) : 0;
    lastTime = time;
    render(delta);
  };

  const syncAnimation = () => {
    window.cancelAnimationFrame(frame);
    frame = 0;
    lastTime = 0;
    if (!narrowScreen.matches && !reducedMotion.matches && !document.hidden) {
      frame = window.requestAnimationFrame(animate);
    }
  };

  resize();
  new ResizeObserver(resize).observe(letterCanvas);
  reducedMotion.addEventListener("change", syncAnimation);
  narrowScreen.addEventListener("change", syncAnimation);
  document.addEventListener("visibilitychange", syncAnimation);
  syncAnimation();
}
