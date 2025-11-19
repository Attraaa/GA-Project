import { Container, Graphics } from "pixi.js";

export class Box extends Container {
  async init() {
    this.pivot.set(-(window.innerWidth / 2), -(window.innerHeight / 2));
    const g = new Graphics();
    g.rect(0, 3, 750, 1073);
    g.stroke({ color: "#ff0000", width: 6 });
    g.pivot.set(g.width / 2, g.height / 2);
    g.eventMode = "static";
    g.cursor = "pointer";

    g.on("pointerup", () => {
      console.log("박스 클릭됨");
      console.log(g.width, g.height);
    });
    this.addChild(g);
    //박스에서 박스를 담을 큰 틀 만들기
    //패키지에서 담을 박스 갯수및 크기 정하기
    //쌓는 코드는 박스에서 패키지에서 다른 클래스 하나 따로 빼야할지????
  }
}
