// subber.js
var zmq = require("zeromq"),
sock = zmq.socket("sub");

sock.connect("tcp://127.0.0.1:5555");
sock.subscribe("1");

sock.on("message", function(message) {
    x = message.toString().slice(2);
    // console.log(
    //     "received a message related to:",
    //     x,
    //     "containing message:",
    //     x
    // );
    let student = JSON.parse(x);
    console.log(student);
});
