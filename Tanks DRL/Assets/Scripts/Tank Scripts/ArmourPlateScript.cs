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

    private void OnCollisionEnter(Collision collision)
    {
        if (tankControllerScript.AITrainer == null) { return; }

        if (collision.gameObject.tag == "Armour Plate") { return; }

        if (collision.gameObject.tag == "round")
        {
            Transform parentComponent = transform.parent;

            while (parentComponent.name != "Hull" || parentComponent.name != "Turret")
            {
                parentComponent = parentComponent.parent;
            }

            tankControllerScript.SendArmourStateData(parentComponent.forward.normalized, collision.GetContact(0).point, collision.gameObject.GetComponent<ShellScript>().penetration - armourThickness, collision.GetContact(0).normal - collision.transform.forward.normalized);
        }
    }
}

