var frame = 0;

function submitForm() {
  var data = document.getElementById("add_website_form");
  data_json = getFormJSON(data);
  insertData(data_json);
}

const getFormJSON = (form) => {
  const data = new FormData(form);
  return Array.from(data.keys()).reduce((result, key) => {
    if (result[key]) {
      result[key] = data.getAll(key);
      return result;
    }
    result[key] = data.get(key);
    return result;
  }, {});
};

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

  document.addEventListener("scroll", function () {
    const maxScroll = document.body.scrollHeight - window.innerHeight;
    const percentScrolled = window.scrollY / maxScroll;
    const translateYValue = -(percentScrolled * 100); // inverts the scroll direction for parallax effect
    document.body.style.setProperty("--translateY", `${translateYValue}%`);
  });

  window.addEventListener("beforeunload", (event) => {
    saveJob();
  });

  const socket = io.connect('/', {
    reconnection: true, // whether to reconnect automatically
    reconnectionAttempts: 5, // number of reconnection attempts before giving up
    reconnectionDelay: 1000, // how long to initially wait before attempting a new reconnection
    reconnectionDelayMax: 5000, // maximum amount of time to wait between reconnection attempts
    randomizationFactor: 0.5,
  });

  socket.on("disconnect", (reason) => {
    console.log("Disconnected: ", reason);
  });

  socket.on("connect_error", (err) => {
    console.log(`connect_error due to ${err.message}`);
  });

  var jobId = -1;
  var start_frame = -1;
  var end_frame = -1;
  var isConnected = false;

  const frameInput = document.getElementById("frameInput");

  //frameInput.addEventListener('blur', sendFrameToServer); // Send when cursor leaves the textbox
  frameInput.addEventListener("keyup", function (event) {
    // Send when "Enter" key is pressed
    if (event.keyCode === 13) {
      sendFrameToServer();
    }
  });
  const jobInput = document.getElementById("job_id");

  jobInput.addEventListener("blur", sendJobToServer);
  jobInput.addEventListener("keyup", function (event) {
    if (event.keyCode === 13) {
      sendJobToServer();
    }
  });


  function frameTrack() {
    let formData = new FormData(document.getElementById("userForm"));
    isConnected = true;
    var data = document.getElementById("userForm");
    let output = "";
    for (let [key, value] of formData.entries()) {
      output += key + ": " + value + "<br>";
    }
    let jsonData = getFormJSON(document.getElementById("userForm"));

    let messageNode = document.getElementById("submit-message");
    messageNode.textContent = "Loading..."

    let img_section = document.getElementById("image-display");
    img_section.style.display = "block";
    //jobId = parseInt(jobInput.value, 10);
    sendJobToServer(jsonData);
  }
  let trackButton = document.getElementById('save_config')
  trackButton.onclick = frameTrack

  function sendJobToServer(formData) {
    jobId = parseInt(jobInput.value, 10);
    if (!isNaN(jobId) && jobId > -1) {
      socket.emit("frame_track_start", formData);
    }
  }


  function saveJob() {
    if (jobId > -1 && isConnected) {
      socket.emit("save_job", jobId);
    }
  }

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

  socket.on("post_annotation", function (response) {
    if (response.type === "text") {
      console.log("Received string:", response.content);
      let infoBox = document.getElementById("output");
      infoBox.innerHTML = "Download Failed";
      // Handle the string data (e.g., display an error message to the user)
    } else if (response.type === "blob") {
      // Handle the blob data (e.g., initiate a download)
      let byteCharacters = atob(response.content);
      let byteNumbers = Array.from(byteCharacters, (char) => char.charCodeAt(0));
      let byteArray = new Uint8Array(byteNumbers);
      const blob = new Blob([byteArray], { type: "application/zip" });
      let url = window.URL.createObjectURL(blob);

      let downloadLink = document.getElementById("downloadLink");
      let downloadDiv = document.getElementById("output");
      downloadDiv.style.display = "none";
      downloadDiv = document.getElementById("downloadDiv");
      downloadDiv.style.display = "block";
      downloadLink.href = url;
      downloadLink.addEventListener("click", function () {
        // Wait for a brief moment before refreshing to ensure the download initiates
        setTimeout(function () {
          location.reload();
        }, 100);
      });

      // Suggest a default filename for the download (optional)
      downloadLink.download = "annotation.zip";

      // Display the download link to the user
      downloadDiv.style.display = "block";
      downloadDiv.className = "download-section";
      downloadLink.style.display = "block";
    }
  });
  socket.on("post_frame_range", function (response) {
    if (response.type === "int") {
      start_frame = response.start_frame;
      end_frame = response.end_frame;
      console.log(
        "got frame range, " +
        "Valid Frame Range is " +
        start_frame +
        " to " +
        end_frame
      );
      var textNode1 = document.createTextNode(
        "Valid Frame Range is " + start_frame + " to " + end_frame
      );
      let section = document.getElementById("frameRange");
      section.appendChild(textNode1);
      section.style.display = "block";
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
  function sendFrameToServer() {
    const frameValue = parseInt(frameInput.value, 10);
    if (!isNaN(frameValue) && jobId > -1 && isConnected) {
      socket.emit("frame_value", {
        job_id: jobId,
        frame: frameValue,
      });
    }
  }


})

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