import { Container, Graphics } from "pixi.js";

export class Box extends Container{

    async init(){
        this.pivot.set(-(window.innerWidth/2),-(window.innerHeight/2))
        const g = new Graphics()
        g.rect(0,3,750,1073)
        g.stroke({color:"#ff0000",width:6})
        g.pivot.set((g.width/2),(g.height/2))
        g.eventMode = "static"
        g.cursor = "pointer"

        g.on("pointerup",()=>{
            console.log("박스 클릭됨")
            console.log(g.width,g.height)
        })
        this.addChild(g)
    }
}