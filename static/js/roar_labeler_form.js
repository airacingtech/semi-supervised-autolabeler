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
  var data = document.getElementById("userForm");
  let output = "";
  for (let [key, value] of formData.entries()) {
    output += key + ": " + value + "<br>";
  }
  document.getElementById("downloadDiv").style.display = "none";
  document.getElementById("output").className = "output-class";
  document.getElementById("output").innerHTML = output;
  let divChild = document.createElement("span");
  divChild.id = "send_button";
  divChild.appendChild(document.createTextNode("Is this input correct?"));
  let button = document.createElement("button");
  button.text = "yes";
  button.textContent = "Yes";
  button.name = "verify_button";
  button.id = "submit_form";

  button.addEventListener("click", function (event) {
    event.preventDefault();
    let divChild = document.getElementById("send_button");
    divChild.append(document.createElement("br"));

    let messageNode = document.createTextNode("Loading...")
    messageNode.id = "message_jobid"
    divChild.appendChild(messageNode);

    fetch(server_address, {
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
      });
  });
  divChild.append(document.createElement("br"));
  divChild.appendChild(button);
  data.appendChild(divChild);
  // return false;  // prevent actual form submission for demonstration
}
async function fetchUpdate() {
  try {
    let response = await fetch("/getUpdate");
    let data = await response.json();
    document.getElementById("content").textContent = data.content;
  } catch (error) {
    console.error("There was an error fetching the update:", error);
  }
}

// Fetch update every 2 seconds
setInterval(fetchUpdate, 2000);
