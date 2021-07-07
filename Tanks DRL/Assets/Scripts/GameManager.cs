using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class GameManager : MonoBehaviour
{
    public float mouseSensitivity = 2f;
    private Camera activeCamera;

    // Start is called before the first frame update
    void Start()
    {
        // Lock the framerate to 60fps
        Application.targetFrameRate = 60;
        QualitySettings.vSyncCount = 0;

        // Lock the cursor inside the game window
        Cursor.lockState = CursorLockMode.Locked;

        // The default camera is the 3rd person camera
        activeCamera = Camera.main;
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    public void SwitchCamera(Camera newCamera)
    {
        activeCamera.gameObject.SetActive(false);
        activeCamera = newCamera;
        activeCamera.gameObject.SetActive(true);

    }
}
