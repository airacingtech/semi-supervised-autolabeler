var frame = 0;

let submitted_manual_track_job_id = -1
let messageNode, img_section, manualTrackJobidNode, manualTrackMessageNode;

let jobId = -1;
let start_frame = -1;
let end_frame = -1;
let isConnected = false;
let socket;



/** Sends data to flask app. */
const insertData = (newData) => {
  console.log("insert data called \n");
  fetch('/upload', {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(newData),
  })
    .then((resp) => resp.json())
    .then((data) => {
      success(data); // display response json from flask to webpage
      console.log(data);
    })
    .catch((error) => console.log(error));
};

$(document).ready(() => {

  messageNode = document.getElementById("submit-message");
  img_section = document.getElementById("image-display");
  manualTrackJobidNode = document.getElementById('manual-track-job-id')
  manualTrackMessageNode = document.getElementById('manual-track-message')

  document.addEventListener("scroll", function () {
    const maxScroll = document.body.scrollHeight - window.innerHeight;
    const percentScrolled = window.scrollY / maxScroll;
    const translateYValue = -(percentScrolled * 100); // inverts the scroll direction for parallax effect
    document.body.style.setProperty("--translateY", `${translateYValue}%`);
  });

  window.addEventListener("beforeunload", (event) => {
    saveJob();
  });

  socket = io.connect('/', {
    reconnection: true, // whether to reconnect automatically
    reconnectionAttempts: 5, // number of reconnection attempts before giving up
    reconnectionDelay: 1000, // how long to initially wait before attempting a new reconnection
    reconnectionDelayMax: 5000, // maximum amount of time to wait between reconnection attempts
    randomizationFactor: 0.5,
  });

  socket.on("connect", (reason) => {
    console.log("Connected: ", reason);
    isConnected = true;
  });

  socket.on("disconnect", (reason) => {
    console.log("Disconnected: ", reason);
    isConnected = false;
  });

  socket.on("connect_error", (err) => {
    console.log(`connect_error due to ${err.message}`);
    isConnected = false;
  });

  const frameInput = document.getElementById("frameInput");

  frameInput.addEventListener("keyup", function (event) {
    // Send when "Enter" key is pressed
    if (event.keyCode === 13) {
      sendFrameToServer();
    }
  });



  const trackButton = document.getElementById('manual_track_button')
  trackButton.onclick = frameTrack


  socket.on("upload_response", function (response) {
    let job_id = response['job_id']
    let divNode = document.getElementById("downloadDiv");
    let linkNode = document.getElementById("downloadLink");
    divNode.style.display = "block";

    if (response['status'] == 'success') {
      divNode.textContent = "completed job " + job_id;
      linkNode.href = "/download-annotation/" + job_id;
      linkNode.style.display = "block";
    } else {
      node.textContent = "error on job " + job_id;
      linkNode.style.display = "none";
    }
  });


  socket.on("post_frame_range", function (response) {
    if (response.type === "int") {
      start_frame = response.start_frame;
      end_frame = response.end_frame;
      let frameRangeNode = document.getElementById("frameRange");
      frameRangeNode.textContent = "Valid frame range: " + start_frame + " to " + end_frame
      frameInput.value = start_frame;
      sendFrameToServer()
    }
  });
  socket.on("post_images", function (response) {
    if (response.type === "image") {
      let img_data = response["img"];
      let img_mask_data = response["img_mask"];
      let img_display1 = document.getElementById("displayImage1");
      let img_display2 = document.getElementById("displayImage2");
      img_display1.src = "data:image/jpeg;base64," + img_data;
      img_display2.src = "data:image/jpeg;base64," + img_mask_data;
    }
  });
})

function sendFrameToServer() {
  const frameValue = parseInt(frameInput.value, 10);
  const jobInput = document.getElementById("job_id");
  jobId = parseInt(jobInput.value, 10);
  if (!isNaN(frameValue) && jobId > -1 && isConnected) {
    socket.emit("frame_value", {
      job_id: jobId,
      frame: frameValue,
    });
  }
}


function sendJobToServer(formData) {
  const jobInput = document.getElementById("job_id");
  jobId = parseInt(jobInput.value, 10);
  if (!isNaN(jobId) && jobId > -1) {
    socket.emit("frame_track_start", formData);
    submitted_manual_track_job_id = jobId
  }
}

function saveJob() {
  if (jobId > -1 && isConnected) {
    socket.emit("save_job", jobId);
    manualTrackMessageNode.textContent = "Check Completed jobs panel to download"
  }
}

function frameTrack() {
  let jsonData = $('form').serializeArray().reduce(function (obj, item) { obj[item.name] = item.value; return obj; }, {});
  messageNode.textContent = "Loading (see below)"
  manualTrackJobidNode.textContent = jsonData['jobId']
  img_section.style.display = "block";
  console.log(jsonData)

  sendJobToServer(jsonData);
}



function backward() {
  changeFrame(-1);
}

function forward() {
  changeFrame(1);
}
function changeFrame(valueChange) {
  const currentFrame = parseInt(frameInput.value, 10);
  if (!isNaN(currentFrame)) {
    frameInput.value = currentFrame + valueChange;
    sendFrameToServer();
  }
}

