import { Application, Graphics } from "pixi.js";

export async function createPixiApp(container: HTMLDivElement | null) {
  // null 체크
  if (!container) {
    console.warn("container is null");
    return;
  }

  const app = new Application();

  await app.init({
    resizeTo: container,
    background: "#1e1e1e",
  });

  container.appendChild(app.canvas);

  // 박스 데이터 스케일 정의
  // Pixi는 픽셀 기준이기 때문에 시각화에서 보기 편하려면 스케일을 지정하여 곱해줘야 보기 편함
  const SCALE = 50;

  // Python에서 사용한 container_width 직접 정의 (pack_boxes.py 기준)
  const CONTAINER_WIDTH = 10;

  // 화면 크기
  const CANVAS_HEIGHT = container.clientHeight;
  const CANVAS_WIDTH = container.clientWidth;

  function drawBox(x: number, y: number, w: number, h: number) {
    const g = new Graphics();

    // 랜덤 색상
    const color = Math.floor(Math.random() * 0xffffff);

    // 바닥 기준 아래에서 위로 쌓는 y 변환
    // Python과 다르게 Pixi는 숫자가 커질 수록 위에서 아래로 내려가기 때문에 변환이 필요
    const pixiY = CANVAS_HEIGHT - (y + h) * SCALE;

    // 박스 전체를 화면 가운데로 정렬
    const totalWidth = CONTAINER_WIDTH * SCALE;
    const offsetX = (CANVAS_WIDTH - totalWidth) / 2;

    // PixiJS v8 방식 rect().fill()
    g.rect(offsetX + x * SCALE, pixiY, w * SCALE, h * SCALE).fill(color);

    //화면에 박스 표시
    app.stage.addChild(g);
  }

  // placements.json 불러오기
  const response = await fetch("/placements.json");
  const data = await response.json();

  // 박스 그리기
  data.forEach((box: any) => {
    drawBox(box.x, box.y, box.width, box.height);
  });

  return app;
}
