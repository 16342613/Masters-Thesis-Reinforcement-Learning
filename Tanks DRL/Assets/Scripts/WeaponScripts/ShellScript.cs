using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ShellScript : MonoBehaviour
{
    // Muzzle velocity in metres per second
    public float muzzleVelocity = 800;
    // Penetrative power in millimetres
    public float penetration = 175;
    // The damage caused by this shell
    public int alphaDamage = 240;
    // The ID of the state before the action which caused the shell to be fired
    public int stateID;
    private ArmourTrainerAI AITrainer;



    private TankControllerScript originTank;

    // Start is called before the first frame update
    void Start()
    {
        Destroy(this.gameObject, 10);
    }

    private void OnCollisionEnter(Collision collision)
    {
        if (collision.gameObject.tag == "Armour Plate")
        {
            TankControllerScript hitTankScript = collision.transform.GetComponentInParent<TankControllerScript>();
            ArmourPlateScript hitArmourScript = collision.transform.GetComponent<ArmourPlateScript>();


            // Start at the current component and work upwards until you reach the hull/turret
            /*
            Transform parentComponent = collision.gameObject.transform;

            while (parentComponent.name != "Hull" || parentComponent.name != "Turret")
            {
                parentComponent = parentComponent.parent;
            }

            TankControllerScript hitTankScript = collision.transform.GetComponentInParent<TankControllerScript>();
            // This may not be necessary if the enemy is stationary!
            hitTankScript.SendArmourStateData(
                parentComponent.forward.normalized, // The main armour forward angle
                collision.GetContact(0).point,  // The global collision contact point
                penetration - collision.gameObject.GetComponent<ArmourPlateScript>().armourThickness,   // The difference in the armour thickness and shell penetration (>0 is better for the shooter)
                collision.GetContact(0).normal - collision.transform.forward.normalized);   // The relative angle of the impact
            */

            // Cause damage if the armour is penned
            if (penetration > hitArmourScript.armourThickness)
            {
                try
                {
                    AITrainer.UpdateReward(stateID, 
                        (1 - ((float)(hitTankScript.GetHitpoints() - alphaDamage) / (float)hitTankScript.GetMaxHitpoints())) * 10f);
                }
                catch (MissingComponentException)
                {
                    Debug.Log("Cant find the AI Armour trainer!");
                }

                hitTankScript.CauseDamage(alphaDamage);
            }
        }

        Destroy(this.gameObject);
    }

    public TankControllerScript Get_originTank()
    {
        return originTank;
    }

    public void SetOriginTank(TankControllerScript originTank)
    {
        this.originTank = originTank;
    }

    public void SetAITrainer(ArmourTrainerAI AITrainer)
    {
        this.AITrainer = AITrainer;
    }
}
