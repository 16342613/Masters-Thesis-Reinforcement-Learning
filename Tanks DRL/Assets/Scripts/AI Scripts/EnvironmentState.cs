using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class EnvironmentState : MonoBehaviour
{
    #region Armour Trainer
    public Vector3 enemyLocation;
    public Vector3 enemyHullAngle;
    public Vector3 enemyTurretAngle;
    public int enemyHitpoints;
    public int enemyMaxHitpoints;
    public int tankIndex;

    public Vector3 shooterLocation;
    public Vector3 shooterForward;
    public float shooterPenetration;
    public int firedRound;
    public Vector3 aimedLocation;
    public float plateThickness;

    public int ID;

    public EnvironmentState(Vector3 enemyLocation, Vector3 enemyHullAngle, Vector3 enemyTurretAngle, int enemyHitpoints, int enemyMaxHitpoints, int tankIndex,
        Vector3 shooterLocation, Vector3 shooterForward, float shooterPenetration, Vector3 aimedLocation, float plateThickness)
    {
        this.enemyLocation = enemyLocation;
        this.enemyHullAngle = enemyHullAngle;
        this.enemyTurretAngle = enemyTurretAngle;
        this.enemyHitpoints = enemyHitpoints;
        this.enemyMaxHitpoints = enemyMaxHitpoints;
        this.tankIndex = tankIndex;

        this.shooterLocation = shooterLocation;
        this.shooterForward = shooterForward;
        this.shooterPenetration = shooterPenetration;
        this.aimedLocation = aimedLocation;
        this.plateThickness = plateThickness;

        this.ID = Random.Range(0, 100000000);
    }
    #endregion

    #region Movement Trainer
    public Vector3 agentPosition;
    public Vector3 targetPosition;

    public EnvironmentState(Vector3 agentPosition, Vector3 targetPosition)
    {
        this.agentPosition = agentPosition;
        this.targetPosition = targetPosition;
    }
    #endregion

    public override string ToString()
    {
        /*return enemyLocation.GetType().ToString() + " | " + enemyHullAngle.GetType().ToString() + " | " + enemyTurretAngle.GetType().ToString() + " | " 
            + enemyHitpoints.GetType().ToString() + " | " + enemyMaxHitpoints.GetType().ToString() + " | " + tankIndex.GetType().ToString() + " | " + shooterLocation.GetType().ToString() + " | " + shooterForward.GetType().ToString() + " | "
            + shooterPenetration.GetType().ToString() + " | " + firedRound.GetType().ToString() + " | " + aimedLocation.GetType().ToString() + " | " + plateThickness.GetType().ToString()

            + " >|< " 
            
            + enemyLocation.ToString() + " | " + enemyHullAngle.ToString() + " | " + enemyTurretAngle.ToString() + " | " + enemyHitpoints.ToString() + " | " + enemyMaxHitpoints.ToString() + " | "
            + tankIndex.ToString() + " | " + shooterLocation.ToString() + " | " + shooterForward.ToString() + " | " + shooterPenetration + " | " + firedRound + " | " + aimedLocation.ToString() + " | " + plateThickness;
        
        return enemyLocation.GetType().ToString() + " | " + shooterLocation.GetType().ToString()

            + " >|< "

            + enemyLocation.ToString() + " | " + shooterLocation.ToString();
        */

        return agentPosition.x.GetType().ToString() + " | " + agentPosition.z.GetType().ToString() + " | " + targetPosition.x.GetType().ToString() + " | " + targetPosition.z.GetType().ToString()

            + " >|< "

            + agentPosition.x.ToString() + " | " + agentPosition.z.ToString() + " | " + targetPosition.x.ToString() + " | " + targetPosition.z.ToString();
    }
}
