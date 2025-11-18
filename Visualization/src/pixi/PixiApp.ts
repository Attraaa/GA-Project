import { Application } from "pixi.js";
import { SceneMain } from "./scenes/SceneMain";

export async function createPixiApp(dom: HTMLElement) {
  const app = new Application();

  await app.init({
    //백그라운드 설정
    background: "#333",
    resizeTo: window,
  });

  dom.appendChild(app.canvas);

  const scene = new SceneMain();
  await scene.load()
  app.stage.addChild(scene);

  return app;
}