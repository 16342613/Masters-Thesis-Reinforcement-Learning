using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ArmourPlateScript : MonoBehaviour
{
    private TankControllerScript tankControllerScript;
    public float armourThickness = 0;

    // Start is called before the first frame update
    void Start()
    {
        tankControllerScript = this.GetComponentInParent<TankControllerScript>();

        if (armourThickness == 0)
        {
            throw new System.Exception("ERROR: The armour thickness of the < " + gameObject.name + " > plate in the < " + transform.root.gameObject.name + " > object is not set!");
        }
    }
}

