(this["webpackJsonpapp-yolov2"]=this["webpackJsonpapp-yolov2"]||[]).push([[0],{165:function(e,t,a){e.exports=a(313)},170:function(e,t,a){},171:function(e,t,a){},172:function(e,t,a){},177:function(e,t,a){},179:function(e,t,a){},210:function(e,t,a){},295:function(e,t,a){},297:function(e,t,a){},313:function(e,t,a){"use strict";a.r(t);var n=a(1),r=a.n(n),o=a(50),c=a.n(o),s=(a(170),a(171),a(31)),l=(a(172),a(22));function i(){return r.a.createElement("div",{className:"Home"},r.a.createElement("div",{className:"lander"},r.a.createElement("img",{src:"logo.jpg",alt:"DEMO object detection app"}),r.a.createElement("p",null,"Detect objects on your images using computer vision"),r.a.createElement(l.Link,{to:"/inference",className:"btn btn-danger btn-lg"},"Get started")))}a(177);var m=a(81),u=a.n(m),d=a(138),p=a(24),h=a(323),b=a(324),g=a(330),f=a(155),E=a(319),v=a(141);a(179);function y(e){var t=e.isLoading,a=e.className,n=void 0===a?"":a,o=e.disabled,c=void 0!==o&&o,s=Object(f.a)(e,["isLoading","className","disabled"]);return r.a.createElement(E.a,Object.assign({className:"LoaderButton ".concat(n),disabled:c||t},s),t&&r.a.createElement(v.a,{glyph:"refresh",className:"spinning"}),s.children)}var w={REGION:"REPLACE_REGION",URL:"https://REPLACE_APIID.execute-api.REPLACE_REGION.amazonaws.com/Production/inference"},j=(a(210),a(327));function k(e){var t=e.toString(),a=t.split("status code ")[1];if("504"==a||"500"==a){return!0}return e instanceof Error||!e.message||(t=e.message),alert(t),!1}var O=a(320),A=a(321),L=a(322);function S(){var e=Object(n.useRef)(null),t=Object(n.useState)(""),a=Object(p.a)(t,2),o=a[0],c=a[1],s=Object(n.useState)(""),l=Object(p.a)(s,2),i=l[0],m=l[1],f=Object(n.useState)(!1),v=Object(p.a)(f,2),w=v[0],S=v[1],x=Object(n.useState)(!1),I=Object(p.a)(x,2),N=I[0],P=I[1],C=Object(n.useState)(""),R=Object(p.a)(C,2),T=R[0],D=R[1];function G(e){return B.apply(this,arguments)}function B(){return(B=Object(d.a)(u.a.mark((function t(a){var n,r;return u.a.wrap((function(t){for(;;)switch(t.prev=t.next){case 0:if(a.preventDefault(),!e.current||"image/png"==e.current.type||"image/jpeg"==e.current.type){t.next=4;break}return alert(" Pick a JPG or PNG file "),t.abrupt("return");case 4:return S(!0),D("Processing..."),t.prev=7,console.log("inference"),t.next=11,J(i);case 11:n=t.sent,c(n),_(),t.next=23;break;case 16:if(t.prev=16,t.t0=t.catch(7),r=k(t.t0),P(!1),!r){t.next=23;break}return D("Just few more seconds, Lambda is warming up..."),t.abrupt("return",G(a));case 23:D(""),S(!1);case 25:case"end":return t.stop()}}),t,null,[[7,16]])})))).apply(this,arguments)}function _(){setTimeout((function(){P(!0)}),5e3)}function J(e){return j.a.post("inference","",{headers:{"Content-Type":"application/json",Accept:"application/json"},body:e})}return r.a.createElement("body",null,r.a.createElement("div",null,N?r.a.createElement(r.a.Fragment,null,r.a.createElement(O.a,{className:"PostInference"},r.a.createElement(A.a,null,r.a.createElement(L.a,{xs:5,md:5},r.a.createElement("h4",null,"Liked it?"),r.a.createElement("h5",null,"If you are curious, learn more about the architecture of this app.")),r.a.createElement(L.a,{xs:5,md:5},r.a.createElement("p",null,r.a.createElement(E.a,{size:"sm",variant:"primary",href:"/architecture"},"Learn more")))))):null),r.a.createElement("div",{className:"NewInference"},r.a.createElement("form",null,r.a.createElement(h.a,{controlId:"imageFile"},r.a.createElement(b.a,null,"Choose an image to process (jpg or png)"),r.a.createElement(g.a,{onChange:function(t){e.current=t.target.files[0];var a=new FileReader;a.readAsDataURL(e.current),a.onloadend=function(){var e=a.result,t=e.substr(e.indexOf(",")+1);m(t)}},type:"file"})),r.a.createElement("p",null,T))),r.a.createElement("div",{className:"Button"},r.a.createElement("form",{onSubmit:G},r.a.createElement(y,{type:"submit",bsSize:"large",bsStyle:"danger",isLoading:w},"Detect objects"))),r.a.createElement("img",{class:"center",src:""==o?"":"data:image/png;base64,".concat(o)}))}a(295);function x(){var e=Object(n.useState)(!0),t=Object(p.a)(e,2),a=(t[0],t[1],Object(n.useState)([])),o=Object(p.a)(a,2),c=o[0];o[1];return r.a.createElement("div",{className:"Home"},r.a.createElement("p",null,"This is my result: ",c))}var I=a(152),N=a(328),P=a(151),C=a.n(P);a(297);function R(){var e={header:["The back-end is built on top of Serverless Node.js Starter, a part of ",r.a.createElement("a",{href:"https://serverless-stack.com"},"Serverless Stack"),", an open-source project developed by Anomaly Innovations.                 They created a step-by-step guide to help you build a full-stack serverless application hosted on AWS. \n                 The front-end is a single page app build on top of ",r.a.createElement("a",{href:"https://github.com/facebook/create-react-app"},"Create React App project")," developed by Facebook."],1:"The browser downloads and stores the static content used by this website.           The user interface was built with React.js, and server-side JavaScript execution is possible thanks to Node.js runtime.           Open-source AWS framework Amplify is used to handle API calls.",2:"With the click of a button the uploaded image is base64 encoded for safe transport through the web           and AWS Amplify makes a REST API call to API Gateway, which is set up as Lambda Proxy integration.",3:["API Gateway invokes Lambda function that hosts Python code used to detect objects on input images.           The inference code uses ",r.a.createElement("a",{href:"https://pjreddie.com/darknet/yolov2/"},"YOLOv2")," , a deep learning model created by Joseph Redmon.           The keras implementation of YOLOv2 I use in this project is one of my assignments from ",r.a.createElement("a",{href:"https://www.coursera.org/specializations/deep-learning?utm_source=gg&utm_medium=sem&utm_content=17-DeepLearning-LATAM&campaignid=6516520287&adgroupid=77982690923&device=c&keyword=coursera%20deep%20learning%20ai&matchtype=b&network=g&devicemodel=&adpostion=&creativeid=383382632097&hide_mobile_promo&gclid=CjwKCAjwkJj6BRA-EiwA0ZVPVg2MCerBH5g0Hh03wK0ESxG68Ty0ulraJbtGfk9VcnZs95aaIdgyrRoCLY4QAvD_BwE"},"Deep Learning Specialization course")," created by deeplearning.ai. The course assignment was greatly inspired by ",r.a.createElement("a",{href:"https://github.com/allanzelener/YAD2K"},"Yet Another Darknet 2 Keras project")," by Allan Zelener.           YOLOv2 detects thousands of potential objects on the input image by specifying bounding box coordinates relative to image dimensions.           Thanks to non-max-suppression technique, Lambda outputs only the most probable predictions."],4:"Due to considerable volume of machine learning libraries and model weights, all Lambda dependencies are stored on Amazon Elastic File System associated with the function.",5:"Lambda returns the provided input image that has been marked with bouding boxes which represent detected objects. Each bounding box features a label that classifies the detected object. For safe web transport to the user, the image is again base64 encoded.",6:"API Gateway via Amplify passes through the response generated by Lambda to the client browser."},t={header:"DEMO object detection app",1:"1 - Static website hosting on S3",2:"2 - API Gateway call via Amplify",3:"3 - API Lambda Proxy integration",4:"4 - Lambda intergrated with EFS for machine learning inference",5:"5 - Lambda integration response",6:"6 - API Gateway response via Amplify"},a=Object(n.useState)(t.header),o=Object(p.a)(a,2),c=o[0],s=o[1],l=Object(n.useState)(e.header),i=Object(p.a)(l,2),m=i[0],u=i[1];return r.a.createElement(I.a,null,r.a.createElement(A.a,{className:"row"},r.a.createElement(L.a,{xs:6,md:4,className:"col1"},r.a.createElement("p",null," Click on numbers on the diagram to learn more about the workflow."),r.a.createElement(N.a,{bsStyle:"danger",className:"panel"},r.a.createElement(N.a.Heading,null,r.a.createElement(N.a.Title,{componentClass:"h3"},c)),r.a.createElement(N.a.Body,null,m))),r.a.createElement(L.a,{xs:6,md:4},r.a.createElement(C.a,{src:"https://github.com/molly-moon/app-object-detection/raw/master/architecture.png",map:{name:"my-map",areas:[{name:"1",shape:"circle",coords:[133,59,10],strokeColor:"red"},{name:"2",shape:"circle",coords:[247,197.5,10],strokeColor:"red"},{name:"3",shape:"circle",coords:[484,198,10],strokeColor:"red"},{name:"4",shape:"circle",coords:[620,331,10],strokeColor:"red"},{name:"5",shape:"circle",coords:[485,259,10],strokeColor:"red"},{name:"6",shape:"circle",coords:[248,259,10],strokeColor:"red"}]},width:794,onClick:function(a){var n=a.name.toString();s(t[n]),u(e[n])}}))))}function T(){return r.a.createElement(s.g,null,r.a.createElement(s.d,{exact:!0,path:"/"},r.a.createElement(i,null)),r.a.createElement(s.d,{exact:!0,path:"/inference"},r.a.createElement(S,null)),r.a.createElement(s.d,{exact:!0,path:"/result"},r.a.createElement(x,null)),r.a.createElement(s.d,{exact:!0,path:"/architecture"},r.a.createElement(R,null)),r.a.createElement(s.d,null,r.a.createElement(i,null)))}var D=a(329),G=a(325),B=a(326),_=a(153);var J=function(){return r.a.createElement("div",{className:"App container"},r.a.createElement(D.a,{fluid:!0,collapseOnSelect:!0},r.a.createElement(D.a.Header,null,r.a.createElement(D.a.Brand,null,r.a.createElement(l.Link,{to:"/"},r.a.createElement("img",{src:"logo.jpg",alt:"DEMO object detection app",width:"130px",height:"50px"}))),r.a.createElement(D.a.Toggle,null)),r.a.createElement(D.a.Collapse,null,r.a.createElement(G.a,{pullRight:!0},r.a.createElement(_.LinkContainer,{to:"/architecture"},r.a.createElement(B.a,null,"How it works"))))),r.a.createElement(T,null))};Boolean("localhost"===window.location.hostname||"[::1]"===window.location.hostname||window.location.hostname.match(/^127(?:\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)){3}$/));a(35).a.configure({API:{endpoints:[{name:"inference",endpoint:w.URL,region:w.REGION}]}}),c.a.render(r.a.createElement(r.a.StrictMode,null,r.a.createElement(l.BrowserRouter,null,r.a.createElement(J,null))),document.getElementById("root")),"serviceWorker"in navigator&&navigator.serviceWorker.ready.then((function(e){e.unregister()})).catch((function(e){console.error(e.message)}))}},[[165,1,2]]]);
//# sourceMappingURL=main.ba4731f1.chunk.js.map