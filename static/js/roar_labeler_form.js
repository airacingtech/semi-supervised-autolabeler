if (!isNaN) {
  isNaN = (value) => {
    return value !== value;
  };
}
var config = null;
function updateForm() {
  let dropdown = document.getElementById("segmentationDropdown");
  let value = dropdown.options[dropdown.selectedIndex].value;
  let framesDiv = document.getElementById("framesDiv");
  if (value === "resegmentation") {
    framesDiv.style.display = "block";
  } else {
    framesDiv.style.display = "none";
  }
}
function saveConfig() {
  var data = document.getElementById("userForm");
  let data_json = getFormJSON(data);
  config = data_json;
}

function onSubmit() {
  let formData = new FormData(document.getElementById("userForm"));

    let messageNode = document.getElementById("submit-message")
    messageNode.textContent = "Submitting job " + formData.get('jobId') + "..."

    fetch('/upload', {
      method: "POST",
      body: formData,
    })
      .then((res) => res.json())
      .then((data) => {
        console.log(data)
        messageNode.textContent = data["message"]
      })
      .catch((error) => {
        console.error("Error:", error);
        messageNode.textContent = "Error submitting job " + formData.get('jobId') + "."
      });
}

async function fetchUpdate() {
  try {
    let response = await fetch("/jobs-status");
    let data = await response.json();
    document.getElementById("uploaded-jobs").textContent = data.ready.join('\n');
    document.getElementById("queued-jobs").textContent = data.queued.join('\n');
    document.getElementById("inprogress-jobs").textContent = data.in_progress.join('\n');

    document.getElementById("completed-jobs").innerHTML = data.done.map(jobid => {
      return `<a target="_blank" href="/download-annotation/${jobid}">${jobid}</a>`
    }).join('\n');
  } catch (error) {
    console.error("There was an error fetching the update:", error);
  } finally {
    setTimeout(fetchUpdate, 2000);
  }
}

$(document).ready(() => {
  fetchUpdate()
})