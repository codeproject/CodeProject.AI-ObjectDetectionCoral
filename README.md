# ObjectDetection (Coral.AI) Module for CodeProject.AI Server

This is an ObjectDetection module using the Google Coral.AI TPU for [CodeProject.AI Server](https://www.codeproject.com/Articles/5322557/CodeProject-AI-Server-AI-the-easy-way). The actual module itself is downloadable via the CodeProject.AI Server's dashboard.

## To develop and debug this code

1. Clone the main [server repo](https://github.com/codeproject/CodeProject.AI-Server) into a directory such as `CodeProject/CodeProject.AI-Server`

2. Clone this ObjectDetectionCoral repo into a separate folder `CodeProject/CodeProject.AI-Modules`

    You should now have

    ```text
    CodeProject
      - CodeProject.AI-Server
         - demos
         - src
         - ... etc
      - CodeProject.AI-Modules
         - ObjectDetectionCoral
         (this repo)
    ```

3. **If you have NOT run dev setup on the server**
    Run the server dev setup scripts by opening a terminal in `CodeProject.AI-Server/src/` then, for Windows, run `setup.bat`, or for Linux/macOS run `bash setup.sh`.<br>
    This will setup the server, and will also setup this module as long as this module sits under a folder named `CodeProject.AI-Modules`, with `CodeProject.AI-Modules` being at the same folder level as `CodeProject.AI-Server`.

    **If you have already setup the server**
    You can run the setup for just this module running the setup script from a terminal opened in this folder
   ```BAT
   REM For Windows
   ..\..\CodeProject.AI-Server\src\setup.bat
   ```
   ```bash
   # For Linux/macOS
   bash ../../CodeProject.AI-Server/src/setup.sh
   ```
4. Open the server repo in Visual Studio Code (or Visual Studio) and build and launch the server (Build and Launch server in the Run and Debug menu in VS Code). This will start the server, which in turn will load the settings file from this module.
    <br>You can start this module directly from the CodeProject.AI Server dashboard, or you can run this module as a separate process via the 'Launch ObjectDetectionCoral' Debug and Run option in VS Code.

## To create a package for this module

Assuming the folder structure outlined above, run

   ```BAT
   REM For Windows
   ..\..\CodeProject.AI-Server\src\create_packages.bat
   ```
   ```bash
   # For Linux/macOS
   bash ../../CodeProject.AI-Server/src/create_packages.sh
   ```
