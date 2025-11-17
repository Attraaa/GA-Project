import { useEffect, useRef } from "react";
import { Application, Assets, Container, Sprite } from "pixi.js";

export default function App() {
  const pixiRef = useRef(null);

  useEffect(() => {
    let app;
    (async () => {
      app = new Application();
      await app.init({
        background: "#1099bb",
        resizeTo: window,
      });

      pixiRef.current.appendChild(app.canvas);

      const container = new Container();
      app.stage.addChild(container);

      const texture = await Assets.load("https://pixijs.com/assets/bunny.png");

      for (let i = 0; i < 25; i++) {
        const bunny = new Sprite(texture);
        bunny.x = (i % 5) * 40;
        bunny.y = Math.floor(i / 5) * 40;
        container.addChild(bunny);
      }

      container.x = app.screen.width / 2;
      container.y = app.screen.height / 2;
      container.pivot.set(container.width / 2, container.height / 2);

      app.ticker.add((time) => {
        container.rotation -= 0.01 * time.deltaTime;
      });
    })();

    return () => {
      if (app) app.destroy(true, true);
    };
  }, []);

  return (
    <div
      ref={pixiRef}
      style={{ width: "100vw", height: "100vh", overflow: "hidden" }}
    />
  );
}