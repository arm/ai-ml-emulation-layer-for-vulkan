Cloning the repository
======================

To clone the |EL_project| as a stand-alone repository, you can use regular git clone commands. However, for
better management of dependencies and to ensure everything is placed in the appropriate directories, we recommend
using the :code:`git-repo` tool to clone the repository as part of the ML SDK for Vulkan® suite. The tool is available here:
(|git_repo_tool_url|).

For a minimal build and to initialize only the |EL_project| and its dependencies, run:

.. code-block:: bash

    repo init -u https://github.com/arm/ai-ml-sdk-manifest -g emulation-layer

Alternatively, to initialize the repo structure for the entire ML SDK for Vulkan®, including the Emulation Layer, run:

.. code-block:: bash

    repo init -u https://github.com/arm/ai-ml-sdk-manifest -g all

After the repo is initialized, you can fetch the contents with:

.. code-block:: bash

    repo sync

.. admonition:: Note: Cloning on Windows®

    To ensure nested submodules do not exceed the maximum long path length, you must enable long paths on Windows®, and
    you must clone close to the root directory or use a symlink. Make sure to use Git for Windows.

    Using **PowerShell**:

    .. code-block:: powershell

        Set-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1
        git config --global core.longpaths true
        git --version # Ensure you are using Git for Windows, for example 2.50.1.windows.1
        git clone <git-repo-tool-url>
        python <path-to-git-repo>\git-repo\repo init -u <manifest-url> -g all
        python <path-to-git-repo>\git-repo\repo sync

    Using **Git Bash**:

    .. code-block:: bash

        cmd.exe "/c reg.exe add \"HKLM\System\CurrentControlSet\Control\FileSystem"" /v LongPathsEnabled /t REG_DWORD /d 1 /f"
        git config --global core.longpaths true
        git --version # Ensure you are using the Git for Windows, for example 2.50.1.windows.1
        git clone <git-repo-tool-url>
        python <path-to-git-repo>/git-repo/repo init -u <manifest-url> -g all
        python <path-to-git-repo>/git-repo/repo sync

Due to a known issue in :code:`git-repo`, nested submodules do not always update as part of :code:`repo sync` and need to
be manually updated, for example:

.. code-block:: bash

    cd dependencies/SPIRV-Tools
    git submodule update --init --recursive

After the sync command completes successfully, you can find the |EL_project| in :code:`<repo_root>/sw/emulation-layer/`.
You can also find all the dependencies required by the |EL_project| in :code:`<repo_root>/dependencies/`.
