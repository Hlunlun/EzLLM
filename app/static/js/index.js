
// fix the header and load content only
$(document).ready(onWindowLoad, false);


function onWindowLoad() {


  $("a.smooth-transition").on("click", function (e) {
    $("page").addClass("fade-out");
    setTimeout(function () {
      $("#page").load(url + " #content");
    }, 500);
  });

  
  // get sequence
  // var sequences = $('textarea[name="seq_field"]');
  uploadFile()


  document.documentElement.style.setProperty('--menu-height', document.getElementById('menu').offsetHeight + 'px');
}

// open other web page after clicking link on table
$(document).ready(function() {
  // Find all <a> elements inside <table> elements and set target="_blank"
  $('table a').attr('target', '_blank');
});

// post to flask 
function sendDataToFlask(data,message) {
  fetch('/post_data', {
      method: 'POST',
      headers: {
          'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        data: data,
        message:message,
      })
  })
  .then(response => response.json())
  .then(result => {
      console.log('Success:', result);
  })
  .catch(error => {
      console.error('Error:', error);
  });
}

// get from flask
function getDataFromFlask() {
  fetch('/get_data')
  .then(response => response.json())
  .then(data => {
      console.log('Data from Flask:', data);
      return data
  })
  .catch(error => console.error('Error:', error));
}


function uploadFile(){
  var dropzone = new Dropzone('#upload-file', {
    previewTemplate: document.querySelector('#preview-template').innerHTML,
    url: "/generation",
    method: "post",
    headers: {
      "Content-Type": "multipart/form-data",
    },
    init: function() {
      this.on("sending", function(file, xhr, formData) {
        formData.append("message", "con_gen"); // 添加message参数
      });
    },
    maxFiles: 1,
    maxFilesize: 512,
    acceptedFiles: ".fasta"
  });  

  
}

