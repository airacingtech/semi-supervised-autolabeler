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

  let data_json = getFormJSON(data);
  button.addEventListener("click", function (event) {
    event.preventDefault();
    let divChild = document.getElementById("send_button");
    divChild.append(document.createElement("br"));
    divChild.appendChild(document.createTextNode("Loading..."));
    // divChild.appendChild(document.createTextNode("Loading..."));

    let jsonData = getFormJSON(document.getElementById("userForm")); // Convert form data to JSON

    fetch(server_address, {
      method: "POST",
      body: formData,
    })
      .then((response) => {
        const contentType = response.headers.get("Content-Type");

        if (contentType && contentType.includes("text")) {
          return response.text(); // process as text
        }
        return response.blob(); // process as blob
      })
      .then((data) => {
        if (typeof data === "string") {
          console.log("Received string:", data);
          let infoBox = document.getElementById("output");
          infoBox.innerHTML = "Download Failed";
          // Handle the string data (e.g., display an error message to the user)
        } else {
          // Handle the blob data (e.g., initiate a download)
          const blob = data;
          let url = window.URL.createObjectURL(blob);

          // Set the download link's href to the object URL
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
      })
      .then((data) => {
        console.log(data);
        // Handle response here
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
