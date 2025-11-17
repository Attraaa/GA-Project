import { Container, Sprite, Assets } from "pixi.js";

export class SceneMain {
    //메인 코딩은 이부분에서 하면됨
  container: Container;

  constructor(app) {
    this.container = new Container();

    this.load(app);
  }

  async load(app) {
    const texture = await Assets.load("https://pixijs.com/assets/bunny.png");

    for (let i = 0; i < 25; i++) {
      const bunny = new Sprite(texture);
      bunny.x = (i % 5) * 40;
      bunny.y = Math.floor(i / 5) * 40;
      this.container.addChild(bunny);
    }

    this.container.x = app.screen.width / 2;
    this.container.y = app.screen.height / 2;
    this.container.pivot.set(this.container.width / 2, this.container.height / 2);

    app.ticker.add(() => {
      this.container.rotation += 0.01;
    });
  }
}