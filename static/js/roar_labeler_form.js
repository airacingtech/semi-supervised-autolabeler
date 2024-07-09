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
var isAdvOptActive = false;
document.getElementById('toggleButton').addEventListener('click', (e) => {
  e.preventDefault();
  isAdvOptActive = !isAdvOptActive;
  toggleFunction(isAdvOptActive);
});

function toggleFunction(state) {
  var adv_opt_div = document.getElementById("advOptDiv");
    if (state) {
        console.log('Toggled On');
        
        adv_opt_div.style.display = "block";
    } else {
        console.log('Toggled Off');
        adv_opt_div.style.display = "none";
    }
}
function onSubmit() {
  let formData = new FormData(document.getElementById("userForm"));

    let messageNode = document.getElementById("submit-message")
    messageNode.textContent = "Submitting job " + formData.get('jobId') + "..."

    fetch('/upload', {
      method: "POST",
      body: formData,
    })
      .then((res) => {
        console.log(res)
        return res.json()})
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
    document.getElementById("uploaded-jobs").textContent = data.ready.map((a)=>a[0]).join('\n');
    document.getElementById("queued-jobs").textContent = data.queued.map((a)=>a[0]).join('\n');
    document.getElementById("inprogress-jobs").textContent = data.in_progress.join('\n');
    document.getElementById("failed-jobs").textContent = data.failed.join('\n');
    document.getElementById("completed-jobs").innerHTML = data.done.map(a=> {
      let jobid = a[0]
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