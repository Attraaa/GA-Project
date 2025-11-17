import { useEffect, useRef } from "react";
import { createPixiApp } from "./pixi/PixiApp";

export default function App() {
  //여긴 걍 로드만 해주도록 했음
  const pixiRef = useRef(null);

  useEffect(() => {
    let app;

    (async () => {
      app = await createPixiApp(pixiRef.current);
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