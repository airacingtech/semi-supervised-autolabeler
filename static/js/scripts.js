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
    var data = document.getElementById('userForm');
    let data_json = getFormJSON(data);
    config = data_json;
}

function onSubmit() {
    let formData = new FormData(document.getElementById('userForm'));
    var data = document.getElementById('userForm')
    let output = "";
    for (let [key, value] of formData.entries()) {
        output += key + ": " + value + "<br>";
    }
    document.getElementById("downloadDiv").style.display = 'none';
    document.getElementById("output").className = 'output-class';
    document.getElementById("output").innerHTML = output;
    let divChild = document.createElement("span")
    divChild.id = "send_button";
    divChild.appendChild(document.createTextNode("Is this input correct?"))
    let button = document.createElement("button")
    button.text = "yes";
    button.textContent = "Yes";
    button.name = "verify_button";
    button.id = "submit_form"
        
    // let data_json = getFormJSON(data);
    button.addEventListener("click", function(event) {
        event.preventDefault(); 
        let divChild = document.getElementById("send_button");
        divChild.append(document.createElement("br"));
        divChild.appendChild(document.createTextNode("Job Sent! Wait for Download Link..."));
        
        // let jsonData = getFormJSON(document.getElementById("userForm")); // Convert form data to JSON
    
        fetch(server_address, {
            method: 'POST',
            body: formData
        })
        .then(response => {
            const contentType = response.headers.get('Content-Type');
        
            if (contentType && contentType.includes('text')) {
              return response.text();  // process as text
            }
            return response.blob();  // process as blob
          })
          .then(data => {
            if (typeof data === 'string') {
                console.log('Received string:', data);
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
                downloadDiv.style.display = 'none';
                downloadDiv = document.getElementById("downloadDiv");
                downloadDiv.style.display = 'block';
                downloadLink.href = url;
                downloadLink.addEventListener('click', function() {
                    // Wait for a brief moment before refreshing to ensure the download initiates
                    setTimeout(function() {
                        location.reload();
                    }, 100);
                });
        
                // Suggest a default filename for the download (optional)
                downloadLink.download = "annotation.zip";
        
                // Display the download link to the user
                downloadDiv.style.display = 'block';
                downloadDiv.className = "download-section";
                downloadLink.style.display = 'block';
            }
          }).then(data => {
            console.log(data);
            // Handle response here
        })
          .catch(error => {
            console.error('Error:', error);
          });
        
    });
    divChild.append(document.createElement("br"))
    divChild.appendChild(button)
    data.appendChild(divChild)
    <!-- return false;  // prevent actual form submission for demonstration -->
}
var isAdvOptActive = false;
document.getElementById('toggleButton').addEventListener('click', function() {
    isAdvOptActive = !isAdvOptions;
    toggleFunction(isAdvOptActive);
});

function toggleFunction(state) {
    if (state) {
        console.log('Toggled On');
    } else {
        console.log('Toggled Off');
    }
}
async function fetchUpdate() {
    try {
        let response = await fetch('/getUpdate');
        let data = await response.json();
        document.getElementById('content').textContent = data.content;
    } catch (error) {
        console.error("There was an error fetching the update:", error);
    }
}

// Fetch update every 2 seconds
setInterval(fetchUpdate, 2000);
var server_address = "http://127.0.0.1:5000/upload"
var server_main = "http://127.0.0.1:5000/"
var frame = 0;
function configureServer() {
let serverAddress = new String(document.getElementById("serverAddress").value);
let formData = document.getElementById("userForm")
// You can save the server address somewhere or update the form's action attribute, etc.
if (!serverAddress.includes("/upload")) {
    if (serverAddress[serverAddress.length - 1] == '/') {
        server_main = serverAddress.slice(0, serverAddress.length - 1);
        serverAddress = serverAddress + "upload";
    } else {
        server_main = serverAddress;
        serverAddress = serverAddress + "/upload";
    }
    
}
server_address = serverAddress;
formData.action= serverAddress;
alert("Configured to server: " + serverAddress);
}
function submitForm(){
        
var data = document.getElementById("add_website_form");
data_json = getFormJSON(data);
insertData(data_json);



}
const getFormJSON = (form) => {
const data = new FormData(form);
return Array.from(data.keys()).reduce((result, key) => {
  if (result[key]) {
    result[key] = data.getAll(key)
    return result
  }
  result[key] = data.get(key);
  return result;
}, {});
};

/** Sends data to flask app. */
const insertData = (newData) => {
console.log("insert data called \n")
fetch(server_address, { 
    method: "POST",
    headers: {
        'Content-Type':'application/json'
    },
    body: JSON.stringify(newData)
})
.then(resp => resp.json())
.then((data) => {
    success(data); // display response json from flask to webpage
    console.log(data)
})
.catch(error => console.log(error))
}

document.addEventListener('scroll', function() {
const maxScroll = document.body.scrollHeight - window.innerHeight;
const percentScrolled = window.scrollY / maxScroll;
const translateYValue = -(percentScrolled * 100); // inverts the scroll direction for parallax effect
document.body.style.setProperty('--translateY', `${translateYValue}%`);
});
    window.addEventListener('beforeunload', (event) => {
        saveJob();
    })

    const socket = io.connect(server_main, {
        reconnection: true,       // whether to reconnect automatically
        reconnectionAttempts: 5,  // number of reconnection attempts before giving up
        reconnectionDelay: 1000,  // how long to initially wait before attempting a new reconnection
        reconnectionDelayMax: 5000, // maximum amount of time to wait between reconnection attempts
        randomizationFactor: 0.5
    });
    socket.on('disconnect', (reason) => {
        console.log('Disconnected: ', reason);
    });
    socket.on("connect_error", (err) => {
        console.log(`connect_error due to ${err.message}`);
      });
    var jobId = -1;
    var start_frame = -1;
    var end_frame = -1;
    const frameInput = document.getElementById('frameInput');
    var isConnected = false;
    //frameInput.addEventListener('blur', sendFrameToServer); // Send when cursor leaves the textbox
    frameInput.addEventListener('keyup', function(event) {
        // Send when "Enter" key is pressed
        if (event.keyCode === 13) {
            sendFrameToServer();
        }
    });
    const jobInput = document.getElementById('job_id');

    jobInput.addEventListener('blur', sendJobToServer);
    jobInput.addEventListener('keyup', function(event) {
        if (event.keyCode === 13) {
            sendJobToServer();
        }
    })
    function frameTrack() {
        let formData = new FormData(document.getElementById('userForm'));
        isConnected = true;
        var data = document.getElementById('userForm')
        let output = "";
        for (let [key, value] of formData.entries()) {
            output += key + ": " + value + "<br>";
        }
        document.getElementById("downloadDiv").style.display = 'none';
        document.getElementById("output").className = 'output-class';
        document.getElementById("output").innerHTML = output;
        let divChild = document.createElement("span")
        divChild.id = "send_button";
        divChild.appendChild(document.createTextNode("Is this input correct?"))
        let button = document.createElement("button")
        button.text = "yes";
        button.textContent = "Yes";
        button.name = "verify_button";
        button.id = "submit_form"
            
        let jsonData = getFormJSON(document.getElementById("userForm"));
        button.addEventListener("click", function(event) {
            event.preventDefault(); 
            let divChild = document.getElementById("send_button");
            divChild.append(document.createElement("br"));
            divChild.appendChild(document.createTextNode("Job Sent! Wait for Frames to Load..."));
            let img_section = document.getElementById("image-display");
            img_section.style.display = 'block';
            //jobId = parseInt(jobInput.value, 10);
            sendJobToServer(jsonData);
            
        });
        divChild.append(document.createElement("br"))
        divChild.appendChild(button)
        data.appendChild(divChild)
    }
    function sendJobToServer(formData) {
        jobId = parseInt(jobInput.value, 10);
        if (!isNaN(jobId) && jobId > -1) {
            
            socket.emit('frame_track_start', formData);
        }
    }      
    function saveJob() {
        if (jobId > -1 && isConnected) {
            socket.emit('save_job', jobId);
        }
    }  
    socket.on('post_annotation', function(response) {
        if (response.type === 'text') {
            console.log('Received string:', response.content);
            let infoBox = document.getElementById("output");
            infoBox.innerHTML = "Download Failed";
            // Handle the string data (e.g., display an error message to the user)
        } else if (response.type === 'blob') {
            // Handle the blob data (e.g., initiate a download)
            let byteCharacters = atob(response.content);
            let byteNumbers = Array.from(byteCharacters, char => char.charCodeAt(0));
            let byteArray = new Uint8Array(byteNumbers);
            const blob = new Blob([byteArray], {type: 'application/zip'});
            let url = window.URL.createObjectURL(blob);
        
            let downloadLink = document.getElementById("downloadLink");
            let downloadDiv = document.getElementById("output");
            downloadDiv.style.display = 'none';
            downloadDiv = document.getElementById("downloadDiv");
            downloadDiv.style.display = 'block';
            downloadLink.href = url;
            downloadLink.addEventListener('click', function() {
                // Wait for a brief moment before refreshing to ensure the download initiates
                setTimeout(function() {
                    location.reload();
                }, 100);
            });
    
            // Suggest a default filename for the download (optional)
            downloadLink.download = "annotation.zip";
    
            // Display the download link to the user
            downloadDiv.style.display = 'block';
            downloadDiv.className = "download-section";
            downloadLink.style.display = 'block';
        }
    });
    socket.on('post_frame_range', function(response) {
        if (response.type === 'int') {
            start_frame = response.start_frame;
            end_frame = response.end_frame;
            console.log("got frame range, " + "Valid Frame Range is " +  start_frame + " to " + end_frame);
            var textNode1 = document.createTextNode("Valid Frame Range is " +  start_frame + " to " + end_frame);
            let section = document.getElementById('frameRange');
            section.appendChild(textNode1);
            section.style.display = "block";
        }
    })
    socket.on('post_images', function(response) {
        if (response.type === 'image') {
            let img_data = response['img'];
            let img_mask_data = response['img_mask'];
            let img_display1 = document.getElementById("displayImage1");
            let img_display2 = document.getElementById("displayImage2");
            img_display1.src = 'data:image/jpeg;base64,' + img_data;
            img_display2.src = 'data:image/jpeg;base64,' + img_mask_data;
        }
    })
    function sendFrameToServer() {
        const frameValue = parseInt(frameInput.value, 10);
        if (!isNaN(frameValue) && jobId > -1 && isConnected) {
            socket.emit('frame_value', {
                job_id: jobId,
                frame: frameValue
            });
        }
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