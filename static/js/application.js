
$(document).ready(function(){
    //connect to the socket server.
    var socket = io.connect('http://' + document.domain + ':' + location.port + '/start');
    console.log('http://' + document.domain + ':' + location.port + '/start');

    //receive details from server
    socket.on('newnumber', function(msg) {
        console.log("Received number" + msg.number);
        //maintain a list of ten numbers
        if (msg.number == 0)
        {
            result_string = '<b> Thumb UP selected </b>';
            setTimeout(function(){
                window.location.href = "/ticket_printing";
            }, 4000);
        }
        else if (msg.number == 1) {
            result_string = '<b> Thumb DOWN selected </b>';
            setTimeout(function(){
                window.location.href = "/";
            }, 4000);
        }
        else if (msg.number == 9){
            result_string = '<b>No fruit Detected</b>';
            setTimeout(function(){
                window.location.href = "/";
            }, 4000);
        }
        else if (msg.number == 10) {
            result_string = '<b>Tomato Detected </b>';
            setTimeout(function(){
                window.location.href = "/thumb_video/Tomato";
            }, 4000);
        } 
        else if (msg.number == 11) {
            result_string = '<b>Apple detected </b>';
            setTimeout(function(){
                window.location.href = "/thumb_video/Apple";
            }, 4000);
        }
        else if (msg.number == 12) {
            result_string = '<b>Banana detected</b>';
            setTimeout(function(){
                window.location.href = "/thumb_video/Banana";
            }, 4000);
        }
        else if (msg.number == 13) {
            result_string = '<b>Mango detected </b>';
            setTimeout(function(){
                window.location.href = "/thumb_video/Mango";
            }, 4000);
        }
        else if (msg.number == 14) {
            result_string = '<b>Several fruits detected </b>';
            setTimeout(function(){
                window.location.href = "/";
            }, 4000);
        }
        else {
            result_string = '<b>No selection found </b>';
            setTimeout(function(){
                window.location.href = "/fruit_manual_choice";
            }, 4000);
        }
        $('#log').html(result_string);
    });

});