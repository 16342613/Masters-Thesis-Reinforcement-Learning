using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class EBR_Wheel_Script : MonoBehaviour
{
    private WheelCollider wheelCollider;

    // Start is called before the first frame update
    void Start()
    {
        wheelCollider = this.GetComponent<WheelCollider>();
    }

    // Update is called once per frame
    void Update()
    {
        if (Input.GetKey(KeyCode.W) == true)
        {
            wheelCollider.motorTorque = 1000;
        }
        //this.transform.position = wheelCollider.center;
    }
}
