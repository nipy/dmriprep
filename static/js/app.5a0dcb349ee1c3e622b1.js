webpackJsonp([1],{"4kYn":function(t,e){},"9M+g":function(t,e){},AqYs:function(t,e,a){t.exports=a.p+"static/img/logo.1966635.svg"},Hp5N:function(t,e){},Jmt5:function(t,e){},JxSv:function(t,e){},Jzq8:function(t,e){},NFPx:function(t,e){},NHnr:function(t,e,a){"use strict";Object.defineProperty(e,"__esModule",{value:!0});var n=a("7+uW"),i=a("e6fC"),r=a.n(i);a("Jmt5"),a("9M+g");n.default.use(r.a);var o={render:function(){var t=this.$createElement,e=this._self._c||t;return e("div",{attrs:{id:"app"}},[e("router-view")],1)},staticRenderFns:[]};var s=a("VU/8")({name:"App"},o,!1,function(t){a("JxSv")},null,null).exports,l=a("/ocq"),c=a("GDE4"),u=a.n(c),h={name:"sprite4d",computed:{row_idx:function(){return Math.floor(this.time/this.M)},col_idx:function(){return this.time%this.M},style:function(){var t={width:this.pix+"px",height:this.pix+"px",zoom:3,"background-position-x":"-"+this.col_idx*this.pix+"px","background-position-y":"-"+this.row_idx*this.pix+"px","background-image":"url('data:image/png;base64,"+this.img+"')",display:"inline-flex",opacity:this.opacity};return this.overlayMode&&(t.position="absolute",t.left="calc(50% - "+this.pix/2+"px)"),t}},data:function(){return{}},methods:{},mounted:function(){},watch:{time:function(){}},props:["M","N","pix","num_slices","img","id","time","overlayMode","opacity"]},d={render:function(){var t=this.$createElement,e=this._self._c||t;return e("span",{staticClass:"mt-3"},[e("div",{staticClass:"sprite",style:this.style,attrs:{id:this.id}})])},staticRenderFns:[]};var p=a("VU/8")(h,d,!1,function(t){a("bFKY")},null,null).exports,m=a("vwbq");window.d3=m;var v={name:"lineChart",data:function(){return{svg:null,g:null,x:null,y:null,xDomain:[0,1],yDomain:[0,1],margin:{top:20,right:20,bottom:30,left:50},lines:[]}},computed:{width:function(){return this.$refs.chart.clientWidth-this.margin.left-this.margin.right},height:function(){return this.$refs.chart.clientHeight-this.margin.top-this.margin.bottom}},watch:{highlightIdx:function(){this.updateHighlightPoints()}},methods:{initAx:function(){var t=m.select("#"+this.id),e=t.append("g").attr("transform","translate("+this.margin.left+", "+this.margin.top+")"),a=m.scaleLinear().range([this.height,0]),n=m.scaleLinear().range([0,this.width]);this.svg=t,this.g=e,this.x=n,this.y=a,this.initXaxis(),this.initYaxis(),this.initHighlightPoints()},initXaxis:function(){this.g.append("g").attr("class","x axis").attr("transform","translate(0, "+this.height+")").call(m.axisBottom(this.x))},updateXaxis:function(){this.x.range([0,this.width]).domain(this.xDomain),m.select("#"+this.id+" .x.axis").call(m.axisBottom(this.x))},initYaxis:function(){this.g.append("g").attr("class","y axis").call(m.axisLeft(this.y)).append("text").attr("fill","#000").attr("transform","rotate(-90)").attr("y",6).attr("dy","0.71em").attr("text-anchor","end").text(this.ylabel)},updateYaxis:function(){this.y.range([this.height,0]).domain(this.yDomain),m.select("#"+this.id+" .y.axis").call(m.axisLeft(this.y))},initHighlightPoints:function(){this.g.append("circle").attr("class","dot-series-1"),this.g.append("circle").attr("class","dot-series-2")},updateHighlightPoints:function(){var t=this;m.select("#"+this.id+" .dot-series-1").attr("r",7).attr("cx",function(){return t.x(t.highlightIdx)}).attr("cy",function(){return t.y(t.data[t.highlightIdx][0])}).attr("fill","steelblue"),m.select("#"+this.id+" .dot-series-2").attr("r",7).attr("cx",function(){return t.x(t.highlightIdx)}).attr("cy",function(){return t.y(t.data[t.highlightIdx][1])}).attr("fill","red")},plotData:function(){var t=this,e=m.min(this.data,function(t,e){return e}),a=m.max(this.data,function(t,e){return e}),n=m.max(this.data,function(t,e){return m.max(t)}),i=m.min(this.data,function(t,e){return m.min(t)});this.xDomain=[e,a],this.yDomain=[i,n],this.updateXaxis(),this.updateYaxis();var r=m.line().x(function(e,a){return t.x(a)}).y(function(e,a){return t.y(e[0])}),o=m.line().x(function(e,a){return t.x(a)}).y(function(e,a){return t.y(e[1])});this.lines=[r,o],this.g.append("path").attr("class","series-1").datum(this.data).attr("fill","none").attr("stroke","steelblue").attr("stroke-linejoin","round").attr("stroke-linecap","round").attr("stroke-width",1.5).attr("d",r),this.g.append("path").attr("class","series-2").datum(this.data).attr("fill","none").attr("stroke","red").attr("stroke-linejoin","round").attr("stroke-linecap","round").attr("stroke-width",1.5).attr("d",o),this.g.selectAll(".outlier").data(this.outlier_indices).enter().append("rect").attr("class","outlier").attr("x",function(e){return t.x(e)-1.5}).attr("y",function(){return t.y(n)}).attr("width","3px").attr("height",function(){return t.height+"px"}).attr("fill","black"),this.svg.on("mousemove",function(){})}},mounted:function(){this.initAx(),this.plotData()},props:["ylabel","xlabel","data","id","highlightIdx","outlier_indices"]},f={render:function(){var t=this.$createElement;return(this._self._c||t)("svg",{ref:"chart",staticClass:"lineChart",attrs:{id:this.id}})},staticRenderFns:[]};var g=a("VU/8")(v,f,!1,function(t){a("4kYn")},null,null).exports,x=a("mtWM"),b=a.n(x),_=a("5dBV"),S=a.n(_),C=a("woOf"),y=a.n(C);var w=function(t){function e(t,e){return t.imageSmoothingEnabled=e,t}var a={};return"boolean"==typeof(a=y()({},{nanValue:!1,smooth:!1,flagValue:!1,colorBackground:"#000000",flagCoordinates:!1,origin:{X:0,Y:0,Z:0},voxelSize:1,affine:!1,heightColorBar:.04,sizeFont:.075,colorFont:"#FFFFFF",nbDecimals:3,crosshair:!1,colorCrosshair:"#0000FF",sizeCrosshair:.9,title:!1,numSlice:!1},t)).affine&&!1===a.affine&&(a.affine=[[a.voxelSize,0,0,-a.origin.X],[0,a.voxelSize,0,-a.origin.Y],[0,0,a.voxelSize,-a.origin.Z],[0,0,0,1]]),a.canvas=document.getElementById(t.canvas),a.context=a.canvas.getContext("2d"),a.context=e(a.context,a.smooth),a.canvasY=document.createElement("canvas"),a.contextY=a.canvasY.getContext("2d"),a.canvasZ=document.createElement("canvas"),a.contextZ=a.canvasZ.getContext("2d"),a.canvasRead=document.createElement("canvas"),a.contextRead=a.canvasRead.getContext("2d"),a.canvasRead.width=1,a.canvasRead.height=1,a.onclick=void 0!==t.onclick?t.onclick:"",a.flagCoordinates?a.spaceFont=.1:a.spaceFont=0,a.sprite=document.getElementById(t.sprite),a.nbCol=a.sprite.width/t.nbSlice.Y,a.nbRow=a.sprite.height/t.nbSlice.Z,a.nbSlice={X:void 0!==t.nbSlice.X?t.nbSlice.X:a.nbCol*a.nbRow,Y:t.nbSlice.Y,Z:t.nbSlice.Z},a.widthCanvas={X:0,Y:0,Z:0},a.heightCanvas={X:0,Y:0,Z:0,max:0},0==a.numSlice&&(a.numSlice={X:Math.floor(a.nbSlice.X/2),Y:Math.floor(a.nbSlice.Y/2),Z:Math.floor(a.nbSlice.Z/2)}),a.coordinatesSlice={X:0,Y:0,Z:0},a.planes={},a.planes.canvasMaster=document.createElement("canvas"),a.planes.contextMaster=a.planes.canvasMaster.getContext("2d"),t.overlay=void 0!==t.overlay&&t.overlay,t.overlay&&(a.overlay={},a.overlay.sprite=document.getElementById(t.overlay.sprite),a.overlay.nbCol=a.overlay.sprite.width/t.overlay.nbSlice.Y,a.overlay.nbRow=a.overlay.sprite.height/t.overlay.nbSlice.Z,a.overlay.nbSlice={X:void 0!==t.overlay.nbSlice.X?t.overlay.nbSlice.X:a.overlay.nbCol*a.overlay.nbRow,Y:t.overlay.nbSlice.Y,Z:t.overlay.nbSlice.Z},a.overlay.opacity=void 0!==t.overlay.opacity?t.overlay.opacity:1),t.colorMap=void 0!==t.colorMap&&t.colorMap,t.colorMap&&(a.colorMap={},a.colorMap.img=document.getElementById(t.colorMap.img),a.colorMap.min=t.colorMap.min,a.colorMap.max=t.colorMap.max,t.colorMap.hide=void 0!==t.colorMap.hide&&t.colorMap.hide,a.colorMap.canvas=document.createElement("canvas"),a.colorMap.context=a.colorMap.canvas.getContext("2d"),a.colorMap.canvas.width=a.colorMap.img.width,a.colorMap.canvas.height=a.colorMap.img.height,a.colorMap.context.drawImage(a.colorMap.img,0,0,a.colorMap.img.width,a.colorMap.img.height,0,0,a.colorMap.img.width,a.colorMap.img.height)),a.getValue=function(t,e){if(!e)return NaN;var a,n,i,r,o;i=e.canvas.width,r=NaN,o=1/0;for(var s=0;s<i;s++)a=e.context.getImageData(s,0,1,1).data,(n=Math.pow(a[0]-t[0],2)+Math.pow(a[1]-t[1],2)+Math.pow(a[2]-t[2],2))<o&&(r=s,o=n);return r*(e.max-e.min)/(i-1)+e.min},a.updateValue=function(){var t={},e=[],n=[];if(a.overlay&&!a.nanValue)try{t.XW=Math.round(a.numSlice.X%a.nbCol),t.XH=Math.round((a.numSlice.X-t.XW)/a.nbCol),a.contextRead.fillStyle="#FFFFFF",a.contextRead.fillRect(0,0,1,1),a.contextRead.drawImage(a.overlay.sprite,t.XW*a.nbSlice.Y+a.numSlice.Y,t.XH*a.nbSlice.Z+a.nbSlice.Z-a.numSlice.Z-1,1,1,0,0,1,1);var i=a.contextRead.getImageData(0,0,1,1).data;e=255==i[0]&&255==i[1]&&255==i[2],a.contextRead.fillStyle="#000000",a.contextRead.fillRect(0,0,1,1),a.contextRead.drawImage(a.overlay.sprite,t.XW*a.nbSlice.Y+a.numSlice.Y,t.XH*a.nbSlice.Z+a.nbSlice.Z-a.numSlice.Z-1,1,1,0,0,1,1),n=0==(i=a.contextRead.getImageData(0,0,1,1).data)[0]&&0==i[1]&&0==i[2],a.voxelValue=e&&n?NaN:a.getValue(i,a.colorMap)}catch(t){console.warn(t.message),rgb=0,a.nanValue=!0,a.voxelValue=NaN}else a.voxelValue=NaN},a.multiply=function(t,e){for(var a=t.length,n=t[0].length,i=(e.length,e[0].length),r=new Array(a),o=0;o<a;++o){r[o]=new Array(i);for(var s=0;s<i;++s){r[o][s]=0;for(var l=0;l<n;++l)r[o][s]+=t[o][l]*e[l][s]}}return r},a.updateCoordinates=function(){var t=a.multiply(a.affine,[[a.numSlice.X+1],[a.numSlice.Y+1],[a.numSlice.Z+1],[1]]);a.coordinatesSlice.X=t[0],a.coordinatesSlice.Y=t[1],a.coordinatesSlice.Z=t[2]},a.init=function(){a.widthCanvas.X=Math.floor(a.canvas.parentElement.clientWidth*(a.nbSlice.Y/(2*a.nbSlice.X+a.nbSlice.Y))),a.widthCanvas.Y=Math.floor(a.canvas.parentElement.clientWidth*(a.nbSlice.X/(2*a.nbSlice.X+a.nbSlice.Y))),a.widthCanvas.Z=Math.floor(a.canvas.parentElement.clientWidth*(a.nbSlice.X/(2*a.nbSlice.X+a.nbSlice.Y))),a.widthCanvas.max=Math.max(a.widthCanvas.X,a.widthCanvas.Y,a.widthCanvas.Z),a.heightCanvas.X=Math.floor(a.widthCanvas.X*a.nbSlice.Z/a.nbSlice.Y),a.heightCanvas.Y=Math.floor(a.widthCanvas.Y*a.nbSlice.Z/a.nbSlice.X),a.heightCanvas.Z=Math.floor(a.widthCanvas.Z*a.nbSlice.Y/a.nbSlice.X),a.heightCanvas.max=Math.max(a.heightCanvas.X,a.heightCanvas.Y,a.heightCanvas.Z),a.canvas.width!=a.widthCanvas.X+a.widthCanvas.Y+a.widthCanvas.Z&&(a.canvas.width=a.widthCanvas.X+a.widthCanvas.Y+a.widthCanvas.Z,a.canvas.height=Math.round((1+a.spaceFont)*a.heightCanvas.max),a.context=e(a.context,a.smooth)),a.sizeFontPixels=Math.round(a.sizeFont*a.heightCanvas.max),a.context.font=a.sizeFontPixels+"px Arial",a.planes.canvasMaster.width=a.sprite.width,a.planes.canvasMaster.height=a.sprite.height,a.planes.contextMaster.globalAlpha=1,a.planes.contextMaster.drawImage(a.sprite,0,0,a.sprite.width,a.sprite.height,0,0,a.sprite.width,a.sprite.height),a.overlay&&(a.planes.contextMaster.globalAlpha=a.overlay.opacity,a.planes.contextMaster.drawImage(a.overlay.sprite,0,0,a.overlay.sprite.width,a.overlay.sprite.height,0,0,a.sprite.width,a.sprite.height)),a.planes.canvasX=document.createElement("canvas"),a.planes.contextX=a.planes.canvasX.getContext("2d"),a.planes.canvasX.width=a.nbSlice.Y,a.planes.canvasX.height=a.nbSlice.Z,a.planes.canvasY=document.createElement("canvas"),a.planes.contextY=a.planes.canvasY.getContext("2d"),a.planes.canvasY.width=a.nbSlice.X,a.planes.canvasY.height=a.nbSlice.Z,a.planes.canvasZ=document.createElement("canvas"),a.planes.contextZ=a.planes.canvasZ.getContext("2d"),a.planes.canvasZ.width=a.nbSlice.X,a.planes.canvasZ.height=a.nbSlice.Y,a.planes.contextZ.rotate(-Math.PI/2),a.planes.contextZ.translate(-a.nbSlice.Y,0),a.updateValue(),a.updateCoordinates(),a.numSlice.X=Math.round(a.numSlice.X),a.numSlice.Y=Math.round(a.numSlice.Y),a.numSlice.Z=Math.round(a.numSlice.Z)},a.draw=function(t,e){var n,i,r={},o={X:"",Y:"",Z:""};switch(o.X=Math.ceil((1-a.sizeCrosshair)*a.nbSlice.X/2),o.Y=Math.ceil((1-a.sizeCrosshair)*a.nbSlice.Y/2),o.Z=Math.ceil((1-a.sizeCrosshair)*a.nbSlice.Z/2),e){case"X":r.XW=a.numSlice.X%a.nbCol,r.XH=(a.numSlice.X-r.XW)/a.nbCol,a.planes.contextX.drawImage(a.planes.canvasMaster,r.XW*a.nbSlice.Y,r.XH*a.nbSlice.Z,a.nbSlice.Y,a.nbSlice.Z,0,0,a.nbSlice.Y,a.nbSlice.Z),a.crosshair&&(a.planes.contextX.fillStyle=a.colorCrosshair,a.planes.contextX.fillRect(a.numSlice.Y,o.Z,1,a.nbSlice.Z-2*o.Z),a.planes.contextX.fillRect(o.Y,a.nbSlice.Z-a.numSlice.Z-1,a.nbSlice.Y-2*o.Y,1)),a.context.fillStyle=a.colorBackground,a.context.fillRect(0,0,a.widthCanvas.X,a.canvas.height),a.context.drawImage(a.planes.canvasX,0,0,a.nbSlice.Y,a.nbSlice.Z,0,(a.heightCanvas.max-a.heightCanvas.X)/2,a.widthCanvas.X,a.heightCanvas.X),a.title&&(a.context.fillStyle=a.colorFont,a.context.fillText(a.title,Math.round(a.widthCanvas.X/10),Math.round(a.heightCanvas.max*a.heightColorBar+.25*a.sizeFontPixels))),a.flagValue&&(value="value = "+S()(a.voxelValue).toPrecision(a.nbDecimals).replace(/0+$/,""),valueWidth=a.context.measureText(value).width,a.context.fillStyle=a.colorFont,a.context.fillText(value,Math.round(a.widthCanvas.X/10),Math.round(a.heightCanvas.max*a.heightColorBar*2+.75*a.sizeFontPixels))),a.flagCoordinates&&(n="x = "+Math.round(a.coordinatesSlice.X),i=a.context.measureText(n).width,a.context.fillStyle=a.colorFont,a.context.fillText(n,a.widthCanvas.X/2-i/2,Math.round(a.canvas.height-a.sizeFontPixels/2)));break;case"Y":a.context.fillStyle=a.colorBackground,a.context.fillRect(a.widthCanvas.X,0,a.widthCanvas.Y,a.canvas.height);for(var s=0;s<a.nbSlice.X;s++){var l=s%a.nbCol,c=(s-l)/a.nbCol;a.planes.contextY.drawImage(a.planes.canvasMaster,l*a.nbSlice.Y+a.numSlice.Y,c*a.nbSlice.Z,1,a.nbSlice.Z,s,0,1,a.nbSlice.Z)}a.crosshair&&(a.planes.contextY.fillStyle=a.colorCrosshair,a.planes.contextY.fillRect(a.numSlice.X,o.Z,1,a.nbSlice.Z-2*o.Z),a.planes.contextY.fillRect(o.X,a.nbSlice.Z-a.numSlice.Z-1,a.nbSlice.X-2*o.X,1)),a.context.drawImage(a.planes.canvasY,0,0,a.nbSlice.X,a.nbSlice.Z,a.widthCanvas.X,(a.heightCanvas.max-a.heightCanvas.Y)/2,a.widthCanvas.Y,a.heightCanvas.Y),a.colorMap&&!a.colorMap.hide&&(a.context.drawImage(a.colorMap.img,0,0,a.colorMap.img.width,1,Math.round(a.widthCanvas.X+.2*a.widthCanvas.Y),Math.round(a.heightCanvas.max*a.heightColorBar/2),Math.round(.6*a.widthCanvas.Y),Math.round(a.heightCanvas.max*a.heightColorBar)),a.context.fillStyle=a.colorFont,label_min=S()(a.colorMap.min).toPrecision(a.nbDecimals).replace(/0+$/,""),label_max=S()(a.colorMap.max).toPrecision(a.nbDecimals).replace(/0+$/,""),a.context.fillText(label_min,a.widthCanvas.X+.2*a.widthCanvas.Y-a.context.measureText(label_min).width/2,Math.round(a.heightCanvas.max*a.heightColorBar*2+.75*a.sizeFontPixels)),a.context.fillText(label_max,a.widthCanvas.X+.8*a.widthCanvas.Y-a.context.measureText(label_max).width/2,Math.round(a.heightCanvas.max*a.heightColorBar*2+.75*a.sizeFontPixels))),a.flagCoordinates&&(a.context.font=a.sizeFontPixels+"px Arial",a.context.fillStyle=a.colorFont,n="y = "+Math.round(a.coordinatesSlice.Y),i=a.context.measureText(n).width,a.context.fillText(n,a.widthCanvas.X+a.widthCanvas.Y/2-i/2,Math.round(a.canvas.height-a.sizeFontPixels/2)));case"Z":a.context.fillStyle=a.colorBackground,a.context.fillRect(a.widthCanvas.X+a.widthCanvas.Y,0,a.widthCanvas.Z,a.canvas.height);for(var u=0;u<a.nbSlice.X;u++){var h=u%a.nbCol,d=(u-h)/a.nbCol;a.planes.contextZ.drawImage(a.planes.canvasMaster,h*a.nbSlice.Y,d*a.nbSlice.Z+a.nbSlice.Z-a.numSlice.Z-1,a.nbSlice.Y,1,0,u,a.nbSlice.Y,1)}a.crosshair&&(a.planes.contextZ.fillStyle=a.colorCrosshair,a.planes.contextZ.fillRect(o.Y,a.numSlice.X,a.nbSlice.Y-2*o.Y,1),a.planes.contextZ.fillRect(a.numSlice.Y,o.X,1,a.nbSlice.X-2*o.X)),a.context.drawImage(a.planes.canvasZ,0,0,a.nbSlice.X,a.nbSlice.Y,a.widthCanvas.X+a.widthCanvas.Y,(a.heightCanvas.max-a.heightCanvas.Z)/2,a.widthCanvas.Z,a.heightCanvas.Z),a.flagCoordinates&&(n="z = "+Math.round(a.coordinatesSlice.Z),i=a.context.measureText(n).width,a.context.fillStyle=a.colorFont,a.context.fillText(n,a.widthCanvas.X+a.widthCanvas.Y+a.widthCanvas.Z/2-i/2,Math.round(a.canvas.height-a.sizeFontPixels/2)))}},a.clickBrain=function(t){var e,n,i,r=a.canvas.getBoundingClientRect(),o=t.clientX-r.left,s=t.clientY-r.top;o<a.widthCanvas.X?(n=Math.round((a.nbSlice.Y-1)*(o/a.widthCanvas.X)),i=Math.round((a.nbSlice.Z-1)*((a.heightCanvas.max+a.heightCanvas.X)/2-s)/a.heightCanvas.X),a.numSlice.Y=Math.max(Math.min(n,a.nbSlice.Y-1),0),a.numSlice.Z=Math.max(Math.min(i,a.nbSlice.Z-1),0)):o<a.widthCanvas.X+a.widthCanvas.Y?(o-=a.widthCanvas.X,e=Math.round((a.nbSlice.X-1)*(o/a.widthCanvas.Y)),i=Math.round((a.nbSlice.Z-1)*((a.heightCanvas.max+a.heightCanvas.X)/2-s)/a.heightCanvas.X),a.numSlice.X=Math.max(Math.min(e,a.nbSlice.X-1),0),a.numSlice.Z=Math.max(Math.min(i,a.nbSlice.Z-1),0)):(o=o-a.widthCanvas.X-a.widthCanvas.Y,e=Math.round((a.nbSlice.X-1)*(o/a.widthCanvas.Z)),n=Math.round((a.nbSlice.Y-1)*((a.heightCanvas.max+a.heightCanvas.Z)/2-s)/a.heightCanvas.Z),a.numSlice.X=Math.max(Math.min(e,a.nbSlice.X-1),0),a.numSlice.Y=Math.max(Math.min(n,a.nbSlice.Y-1),0)),a.updateValue(),a.updateCoordinates(),a.drawAll(),a.onclick&&a.onclick(t)},a.drawAll=function(){a.draw(a.numSlice.X,"X"),a.draw(a.numSlice.Y,"Y"),a.draw(a.numSlice.Z,"Z")},a.canvas.addEventListener("click",a.clickBrain,!1),a.canvas.addEventListener("mousedown",function(t){a.canvas.addEventListener("mousemove",a.clickBrain,!1)},!1),a.canvas.addEventListener("mouseup",function(t){a.canvas.removeEventListener("mousemove",a.clickBrain,!1)},!1),a.sprite.addEventListener("load",function(){a.init(),a.drawAll()}),a.overlay&&a.overlay.sprite.addEventListener("load",function(){a.init(),a.drawAll()}),a.init(),a.drawAll(),a},M={name:"brainsprite",props:["base","overlay","id","base_dim_x","base_dim_y","overlay_dim_x","overlay_dim_y"],data:function(){return{brain:null,showOrig:!0,done:!1,ready:!1}},methods:{initBrainSprite:function(){var t=new w({canvas:this.id,sprite:this.id+"_spriteImg",nbSlice:{Y:this.base_dim_x,Z:this.base_dim_y},overlay:{sprite:this.id+"_overlayImg",nbSlice:{Y:this.overlay_dim_x,Z:this.overlay_dim_y},opacity:.5}});this.brain=t,this.showOrig=!1,this.done=!0}},mounted:function(){var t=this;this.$nextTick(function(){setTimeout(function(){t.ready=!0,t.initBrainSprite()},100)})}},X={render:function(){var t=this,e=t.$createElement,a=t._self._c||e;return a("div",{staticClass:"brainsprite"},[t.done?t._e():a("div",[t._v("Hold on...")]),t._v(" "),a("canvas",{attrs:{id:t.id}}),t._v(" "),t.showOrig?a("img",{staticClass:"hidden",attrs:{id:t.id+"_spriteImg",src:"data:image/png;base64,"+t.base}}):t._e(),t._v(" "),t.overlay&&t.showOrig?a("img",{staticClass:"hidden",attrs:{id:t.id+"_overlayImg",src:"data:image/png;base64,"+t.overlay}}):t._e()])},staticRenderFns:[]};var Y=a("VU/8")(M,X,!1,function(t){a("lyc9")},"data-v-b1edb06c",null).exports,Z={name:"report",components:{sprite4d:p,vueSlider:u.a,lineChart:g,BrainSprite:Y},data:function(){return{time:0,spriteSlice:0,report:null}},methods:{get_mid_slice:function(){return Math.floor(this.report.b0.num_slices/2)}},created:function(){var t=this;this.$route.query&&(this.$route.query.url||"Report"!==this.$route.name?this.$route.query.url&&b.a.get(this.$route.query.url).then(function(e){t.report=e.data}):this.$router.push("/"))},mounted:function(){var t=this;this.reportProp&&(this.report=this.reportProp),this.$nextTick(function(){t.report&&(t.spriteSlice=t.get_mid_slice())})},watch:{reportProp:function(){this.reportProp&&(this.report=this.reportProp)},report:function(){this.report&&(this.spriteSlice=this.get_mid_slice())},$route:function(){var t=this;this.$route.query&&(this.$route.query.url||"Report"!==this.$route.name?(console.log("getting axios?",this.$route.query.url),b.a.get(this.$route.query.url).then(function(e){t.report=e.data})):this.$router.push("/"))}},props:{reportProp:Object}},R={render:function(){var t=this,e=t.$createElement,a=t._self._c||e;return a("div",{staticClass:"mt-3 container"},[t.report?a("div",[a("h2",{staticClass:"mt-3 pt-3"},[t._v("Corrected dwi")]),t._v(" "),a("p",{staticClass:"lead"},[t._v("Result of eddy")]),t._v(" "),t._l(t.report.dwi_corrected,function(e){return a("sprite4d",{key:e.orientation,attrs:{M:e.M,N:e.N,img:e.img,num_slices:e.num_slices,pix:e.pix,id:e.orientation,time:t.time,overlayMode:!1,opacity:"1"}})}),t._v(" "),a("vue-slider",{ref:"timeSlider",attrs:{min:0,max:t.report.dwi_corrected[0].num_slices-1},model:{value:t.time,callback:function(e){t.time=e},expression:"time"}}),t._v(" "),a("h2",{staticClass:"mt-3 pt-3"},[t._v("Eddy Report")]),t._v(" "),a("p",{staticClass:"lead"},[a("b-btn",{directives:[{name:"b-toggle",rawName:"v-b-toggle.collapse1",modifiers:{collapse1:!0}}],attrs:{variant:"primary"}},[t._v("\n      Outliers ("+t._s(t.report.eddy_report.length)+")")])],1),t._v(" "),a("b-collapse",{staticClass:"mt-2",attrs:{id:"collapse1"}},[a("b-card",t._l(t.report.eddy_report,function(e){return a("p",{key:e},[t._v(t._s(e))])}),0)],1),t._v(" "),a("div",{staticStyle:{height:"200px",width:"100%",display:"inline-flex"}},[a("line-chart",{attrs:{id:"motion_params",data:t.report.eddy_params,outlier_indices:t.report.outlier_volumes,xlabel:"TR",ylabel:"RMS",highlightIdx:t.time}})],1),t._v(" "),a("h2",{staticClass:"mt-3 pt-3"},[t._v("Registration + Brain Mask")]),t._v(" "),a("p",{staticClass:"lead"},[t._v("Brain mask computed on T1w, and mapped to B0")]),t._v(" "),a("BrainSprite",{ref:"brainMaskSprite",attrs:{id:"brainMaskSprite",base_dim_x:t.report.b0.pix,base_dim_y:t.report.b0.pix,overlay_dim_x:t.report.anat_mask.pix,overlay_dim_y:t.report.anat_mask.pix,base:t.report.b0.img,overlay:t.report.anat_mask.img}}),t._v(" "),a("h2",{staticClass:"mt-3 pt-3"},[t._v("DTI: ColorFA")]),t._v(" "),a("p",{staticClass:"lead"},[t._v("Color FA mapped on B0")]),t._v(" "),a("BrainSprite",{ref:"colorFASprite",attrs:{id:"colorFASprite",base_dim_x:t.report.b0.pix,base_dim_y:t.report.b0.pix,overlay_dim_x:t.report.colorFA.pix,overlay_dim_y:t.report.colorFA.pix,base:t.report.b0.img,overlay:t.report.colorFA.img}})],2):t._e()])},staticRenderFns:[]};var k=a("VU/8")(Z,R,!1,function(t){a("Hp5N")},null,null).exports,F={name:"HelloWorld",components:{sprite4d:p,vueSlider:u.a,lineChart:g,report:k},data:function(){return{file:null,msg:"Welcome to Your Vue.js App",report:{},time:0,spriteSlice:0,url:null,bucket:null}},methods:{get_mid_slice:function(){return Math.floor(this.report.b0.num_slices/2)},navigate:function(){this.$router.push({path:"/report",query:{url:this.url}})},s3:function(){this.$router.push({path:"/bucket/"+this.bucket})}},watch:{file:function(){if(this.file){var t=new FileReader,e=this;t.onload=function(t){var a=t.target.result;e.report=JSON.parse(a)},t.readAsText(this.file)}},report:function(){this.report.b0&&(this.spriteSlice=this.get_mid_slice())}}},E={render:function(){var t=this,e=t.$createElement,n=t._self._c||e;return n("b-container",[n("img",{staticClass:"logo",attrs:{src:a("AqYs")}}),t._v(" "),n("h1",[t._v("dmriprep Viewer")]),t._v(" "),n("p",{staticClass:"lead"},[t._v("Upload your report.json file from dmriprep")]),t._v(" "),n("b-form-file",{staticClass:"mt-3",attrs:{state:Boolean(t.file),placeholder:"Choose a file..."},model:{value:t.file,callback:function(e){t.file=e},expression:"file"}}),t._v(" "),n("p",{staticClass:"lead mt-3"},[t._v("OR copy/paste a URL")]),t._v(" "),n("b-input-group",{staticClass:"mb-3",attrs:{size:"md",prepend:"URL"}},[n("b-form-input",{model:{value:t.url,callback:function(e){t.url=e},expression:"url"}}),t._v(" "),n("b-input-group-append",[n("b-btn",{attrs:{size:"md",text:"Button",variant:"primary"},on:{click:t.navigate}},[t._v("Go")])],1)],1),t._v(" "),n("p",{staticClass:"lead mt-3"},[t._v("OR point to an S3 bucket")]),t._v(" "),n("b-input-group",{staticClass:"mb-3",attrs:{size:"md",prepend:"Bucket"}},[n("b-form-input",{model:{value:t.bucket,callback:function(e){t.bucket=e},expression:"bucket"}}),t._v(" "),n("b-input-group-append",[n("b-btn",{attrs:{size:"md",text:"Button",variant:"primary"},on:{click:t.s3}},[t._v("Go")])],1)],1),t._v(" "),t.report.b0?n("report",{attrs:{reportProp:t.report}}):t._e()],1)},staticRenderFns:[]};var z=a("VU/8")(F,E,!1,function(t){a("XaoR")},"data-v-57ece340",null).exports,B=a("Xxa5"),I=a.n(B),$=a("exGp"),q=a.n($),A=a("M4fF"),P=a.n(A),T=a("vwbq"),N={name:"GroupStats",props:["data","individual"],data:function(){return{}},methods:{scaleZ:function(t){return T.scaleLinear().range([0,100]).domain([-3,3])(t)},numberFormatter:function(t){return T.format(".3n")(t)},getColor:function(t,e){if(t){if(e<=-1)return"danger";if(e>=1)return"success"}return e<=-1?"success":e>=1?"danger":"primary"}},computed:{mean_group_abs_mot:function(){return T.mean(this.data,function(t){return t.qc_mot_abs})},std_group_abs_mot:function(){return T.deviation(this.data,function(t){return t.qc_mot_abs})},individual_abs_mot_z:function(){return this.individual?(this.individual.qc_mot_abs-this.mean_group_abs_mot)/this.std_group_abs_mot:null},mean_group_rel_mot:function(){return T.mean(this.data,function(t){return t.qc_mot_rel})},std_group_rel_mot:function(){return T.deviation(this.data,function(t){return t.qc_mot_rel})},individual_rel_mot_z:function(){return this.individual?(this.individual.qc_mot_rel-this.mean_group_rel_mot)/this.std_group_rel_mot:null}}},V={render:function(){var t=this,e=t.$createElement,a=t._self._c||e;return a("div",[a("div",{staticClass:"mb-3"},[a("p",{staticClass:"lead"},[t._v("\n      Comparison to Group Eddy_Quad Statistics\n    ")]),t._v(" "),a("p",[t._v("Absolute Motion (Z = "+t._s(t.numberFormatter(t.individual_abs_mot_z))+")")]),t._v(" "),a("b-progress",{staticClass:"w-50 mx-auto"},[a("b-progress-bar",{attrs:{value:t.scaleZ(t.individual_abs_mot_z),variant:t.getColor(0,t.individual_abs_mot_z)}},[t._v("\n        z = "+t._s(t.numberFormatter(t.individual_abs_mot_z))+"\n      ")])],1)],1),t._v(" "),a("div",{staticClass:"mb-3"},[a("p",[t._v("Relative Motion (Z = "+t._s(t.numberFormatter(t.individual_rel_mot_z))+")")]),t._v(" "),a("b-progress",{staticClass:"w-50 mx-auto"},[a("b-progress-bar",{attrs:{value:t.scaleZ(t.individual_rel_mot_z),variant:t.getColor(0,t.individual_rel_mot_z)}},[t._v("\n        z = "+t._s(t.numberFormatter(t.individual_rel_mot_z))+"\n      ")])],1)],1)])},staticRenderFns:[]};var H={name:"bucket",data:function(){return{manifestEntries:[],currentReportIdx:0,currentReport:{},ready:!1,allReports:[],statsReady:!1}},components:{Report:k,GroupStats:a("VU/8")(N,V,!1,function(t){a("Jzq8")},null,null).exports},methods:{getReport:function(t){return b.a.get("https://s3-us-west-2.amazonaws.com/"+this.bucket+"/"+t)},getAllReports:function(){var t=this;return q()(I.a.mark(function e(){var a;return I.a.wrap(function(e){for(;;)switch(e.prev=e.next){case 0:return t.statsReady=!1,e.next=3,P.a.map(t.manifestEntries,function(e){return t.getReport(e)});case 3:a=e.sent,P.a.map(a,function(e){e.then(function(e){e.data.eddy_quad&&t.allReports.push(e.data.eddy_quad)})}),t.statsReady=!0;case 6:case"end":return e.stop()}},e,t)}))()},xmlParser:function(t){return(new DOMParser).parseFromString(t,"text/xml")},parseS3:function(t){var e=this.xmlParser(t),a=e.getElementsByTagName("Key"),n=e.getElementsByTagName("NextContinuationToken"),i=e.getElementsByTagName("IsTruncated")[0].innerHTML;this.continuation="true"===i?encodeURIComponent(n[0].innerHTML):null;var r=P.a.map(a,function(t){return t.innerHTML}),o=P.a.filter(r,function(t){return t.endsWith("_report.json")});return P.a.uniq(o)},S3Continuation:function(t){var e=this,a="https://s3-us-west-2.amazonaws.com/"+this.bucket+"/?list-type=2&";return a+="&max-keys=100000",a+="&continuation-token="+t,t?b.a.get(a).then(function(t){var a=e.parseS3(t.data);e.manifestEntries=P.a.uniq(e.manifestEntries.concat(a)),e.continuation&&e.S3Continuation(e.continuation)}):0},getS3Manifest:function(){var t=this,e="https://s3-us-west-2.amazonaws.com/"+this.bucket+"/?list-type=2&";return e+="&max-keys=100000",b.a.get(e).then(function(e){var a=t.parseS3(e.data);t.manifestEntries=P.a.uniq(t.manifestEntries.concat(a)),t.continuation&&t.S3Continuation(t.continuation)})},updateReport:function(){var t=this;this.ready=!1;var e="https://s3-us-west-2.amazonaws.com/"+this.bucket+"/"+this.manifestEntries[this.currentReportIdx];return b.a.get(e).then(function(a){t.currentReport=a.data,t.ready=!0,t.$router.replace({name:"Bucket",params:{bucket:t.bucket},query:{report:e}})})}},computed:{bucket:function(){return this.$route.params.bucket}},watch:{bucket:function(){this.getS3Manifest()},currentReportIdx:function(){this.updateReport()}},mounted:function(){var t=this,e=null;this.$route.query&&this.$route.query.report&&(e=this.$route.query.report.split("https://s3-us-west-2.amazonaws.com/"+this.bucket+"/")[1]),this.getS3Manifest().then(this.updateReport).then(function(){e&&(t.currentReportIdx=t.manifestEntries.indexOf(e)),t.getAllReports()})}},D={render:function(){var t=this,e=t.$createElement,a=t._self._c||e;return a("div",{staticClass:"bucket"},[a("div",{staticClass:"row"},[a("div",{staticClass:"col-md-3"},[a("div",{staticClass:"container"},[a("h3",[t._v(t._s(t.bucket)+" ("+t._s(t.manifestEntries.length)+")")]),t._v(" "),a("div",{staticClass:"mb-3"},[t.statsReady?t._e():a("b-button",{on:{click:t.getAllReports}},[t._v("\n            Compute Statistics\n          ")])],1),t._v(" "),a("b-nav",{staticClass:"w-100",attrs:{vertical:"",pills:""}},t._l(t.manifestEntries,function(e,n){return a("b-nav-item",{key:e,attrs:{active:n===t.currentReportIdx},on:{click:function(e){t.currentReportIdx=n}}},[t._v("\n            "+t._s(e.split("/")[0])+"\n          ")])}),1)],1)]),t._v(" "),a("div",{staticClass:"col-md-9"},[t.manifestEntries.length?a("h1",[t._v("\n        "+t._s(t.manifestEntries[t.currentReportIdx].split("/")[0])+"\n      ")]):t._e(),t._v(" "),t.statsReady?a("div",[a("GroupStats",{attrs:{data:t.allReports,individual:t.currentReport.eddy_quad}})],1):t._e(),t._v(" "),t.ready?a("div",[a("report",{attrs:{reportProp:t.currentReport}})],1):a("div",[t._v("\n        loading...\n      ")])])])])},staticRenderFns:[]};var W=a("VU/8")(H,D,!1,function(t){a("NFPx")},null,null).exports;n.default.use(l.a);var L=new l.a({routes:[{path:"/",name:"HelloWorld",component:z},{path:"/report",name:"Report",component:k},{path:"/bucket/:bucket",name:"Bucket",component:W}]});n.default.config.productionTip=!1,new n.default({el:"#app",router:L,components:{App:s},template:"<App/>"})},XaoR:function(t,e){},bFKY:function(t,e){},lyc9:function(t,e){}},["NHnr"]);
//# sourceMappingURL=app.5a0dcb349ee1c3e622b1.js.map