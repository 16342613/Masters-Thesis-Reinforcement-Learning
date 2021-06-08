using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class GameManager : MonoBehaviour
{
    public float mouseSensitivity = 2f;

    // Start is called before the first frame update
    void Start()
    {
        // Lock the framerate to 60fps
        Application.targetFrameRate = 600;
        QualitySettings.vSyncCount = 0;

        Cursor.lockState = CursorLockMode.Locked;
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
