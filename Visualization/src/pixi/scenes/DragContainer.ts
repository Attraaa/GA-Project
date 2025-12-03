// import { Container, FederatedPointerEvent, Point } from "pixi.js";

// /**
//  * DragContainer 드래그를 할수있는 pixi Container입니다.
//  * dragStart, dragEnd이벤트를 호출합니다.
//  * onDragStart,onDragEnd를 이용하여 콜백을 override하여 사용할수 있습니다.
//  * @event dragStart
//  * @event dragEnd
//  * @author kwonth1210
//  */
// export abstract class DragContainer extends Container {
//   private mDragOffset = { x: 0, y: 0 };

//   constructor() {
//     super();

//     this.eventMode = "static";
//     this.cursor = "pointer";
//     this.on("pointerdown", (evt: FederatedPointerEvent) => {
//       const pos = this.toLocal(evt.global);

//       this.startDragAction(pos, evt);
//     });
//   }

//   startDragAction(offsetPos: Point, _evt: FederatedPointerEvent | null = null) {
//     this.emit("dragStart", _evt);
//     this.mDragOffset.x = offsetPos.x;
//     this.mDragOffset.y = offsetPos.y;
//     //this.mResetPos.x = this.x;
//     //this.mDragStartPos.y = this.y;
//     this.onDragStart(_evt);
//     SparkApp.stage.eventMode = "static";
//     const onMoveProc = (evt: FederatedPointerEvent) => {
//       this.emit("dragMove", evt);
//       const pos = this.parent.toLocal(evt.global);
//       this.position.set(pos.x - this.mDragOffset.x, pos.y - this.mDragOffset.y);
//     };

//     const onUpProc = (evt: any) => {
//       this.emit("dragEnd", evt);
//       this.onDragEnd(evt);
//     };

//     this.once("pointerup", onUpProc);
//     this.once("pointerupoutside", onUpProc);
//   }
//   /**
//    * 드래그 시작시 호출되는 overriding용 콜백함수 입니다.
//    * @param _evt pixi FederatedPointerEvent 입니다.
//    */
//   onDragStart(_evt: FederatedPointerEvent | null): void {}

//   /**
//    * 드래그 종료시 호출되는 overriding용 콜백함수 입니다.
//    * @param _evt pixi FederatedPointerEvent 입니다.
//    */
//   onDragEnd(_evt: FederatedPointerEvent | null): void {}
// }
