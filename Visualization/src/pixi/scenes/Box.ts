import { Container, Graphics } from "pixi.js";

export class Box extends Container {
  w: number;
  h: number;
  positionX: number;
  positionY: number;
  mInfo!: any

  constructor(w: number, h: number, px: number, py: number,info:any) {
    super();
    this.w = w;
    this.h = h;
    this.positionX = px;
    this.positionY = py;
    this.mInfo = info
  }

  init() {
    const g = new Graphics();

    g.rect(0, 0, this.w, this.h);
    g.fill({ color: 0x1e90ff, alpha: 0.55 });
    g.stroke({ color: 0xffffff, width: 2 });

    this.x = this.positionX;
    this.y = this.positionY;

    this.addChild(g);

    // 클릭 테스트
    this.eventMode = "static";
    this.cursor = "pointer";
    this.on("pointerup", () => {
      console.log("클릭된 박스 GEN: ",this.mInfo.id)
      console.log("클릭된 박스 W: ",this.mInfo.w,", H: ",this.mInfo.h)
      console.log("클릭된 박스 xPos: ",this.mInfo.x,", yPos: ",this.mInfo.y)
      console.log("회전 유무: ",this.mInfo.is_rotated)
      console.log("=======================================================")
    });
  }
}