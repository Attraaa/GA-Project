import { Container } from "pixi.js";
import { Box } from "./Box";

export class SceneMain extends Container {
  constructor() {
    super();
  }
  mInfo!: any;

  // 중요!!!!!!!!!!!!!!!!!!!!
  // results 폴더 무조건 public 으로 넣어야됨
  async load() {
    const data = await this.loadJSON(59); //여기 괄호에 제이슨 파일 gen 숫자 넣으면 됨 ㅇㅇ
    //data.placement[] 배열로 넘어옴.
    console.log(data);
    this.mInfo = data;
    this.spawnBoxes();
  }

  async loadJSON(gen: number): Promise<any> {
    const res = await fetch(`/results/gen_${gen}.json`);
    return await res.json();
  }

  spawnBoxes() {
    const containerWidth = this.mInfo.container_width;
    const maxHeight = this.mInfo.max_height;

    const windowWidth = window.innerWidth;
    const windowHeight = window.innerHeight;

    const scaleX = windowWidth / containerWidth;
    const scaleY = windowHeight / maxHeight;
    const scale = Math.min(scaleX, scaleY);

    for (const info of this.mInfo.placements) {
      let boxWidth = info.w;
      let boxHeight = info.h;

      if (info.is_rotated == true) {
        let temp = 0;
        temp = boxWidth;
        boxWidth = boxHeight;
        boxHeight = boxWidth;
      }

      const posX = info.x * scale;
      const posY = info.y * scale;

      const box = new Box(boxWidth * scale, boxHeight * scale, posX, posY, info);
      box.init();
      this.addChild(box);
    }

    const totalWidth = containerWidth * scale;
    const totalHeight = maxHeight * scale;

    this.x = (windowWidth - totalWidth) / 2;
    this.y = (windowHeight - totalHeight) / 2;
  }
}
