using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class EnvironmentState
{
    #region Armour Trainer
    public Vector3 enemyLocation;
    public Vector3 enemyHullAngle;
    public Vector3 enemyTurretAngle;
    public float enemyHitpoints;
    public float enemyMaxHitpoints;
    public int tankIndex;

    public Vector3 shooterLocation;
    public Vector3 shooterForward;
    public Vector3 idealForward;
    public float shooterPenetration;
    public int firedRound;
    public Vector3 aimedLocation;
    public float plateThickness;

    public int ID;

    public EnvironmentState(Vector3 enemyLocation, Vector3 enemyHullAngle, Vector3 enemyTurretAngle, float enemyHitpoints, float enemyMaxHitpoints, int tankIndex,
        Vector3 shooterLocation, Vector3 shooterForward, Vector3 idealForward, float shooterPenetration, Vector3 aimedLocation, float plateThickness)
    {
        this.enemyLocation = enemyLocation;
        this.enemyHullAngle = enemyHullAngle;
        this.enemyTurretAngle = enemyTurretAngle;
        this.enemyHitpoints = enemyHitpoints;
        this.enemyMaxHitpoints = enemyMaxHitpoints;
        this.tankIndex = tankIndex;

        this.shooterLocation = shooterLocation;
        this.shooterForward = shooterForward;
        this.idealForward = idealForward;
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
        #region Armour Trainer ToString

        /*return enemyLocation.GetType().ToString() + " | " + enemyHullAngle.GetType().ToString() + " | " + enemyTurretAngle.GetType().ToString() + " | " 
            + enemyHitpoints.GetType().ToString() + " | " + tankIndex.GetType().ToString() + " | " + shooterLocation.GetType().ToString() + " | " + shooterForward.GetType().ToString() + " | "
            + shooterPenetration.GetType().ToString() + " | " + aimedLocation.GetType().ToString() + " | " + plateThickness.GetType().ToString()

            + " >|< " 
            
            + enemyLocation.ToString() + " | " + enemyHullAngle.ToString() + " | " + enemyTurretAngle.ToString() + " | " + enemyHitpoints.ToString() + " | "
            + tankIndex.ToString() + " | " + shooterLocation.ToString() + " | " + shooterForward.ToString() + " | " + shooterPenetration + " | " + aimedLocation.ToString() + " | " + plateThickness;
        */

        return enemyLocation.GetType().ToString() + " | "
           + enemyHitpoints.GetType().ToString() + " | " + shooterLocation.GetType().ToString() + " | " + shooterForward.GetType().ToString() + " | " + idealForward.GetType().ToString() + " | "
           + aimedLocation.GetType().ToString()

           + " >|< "

           + enemyLocation.ToString() + " | " + enemyHitpoints.ToString() + " | "
           + shooterLocation.ToString() + " | " + shooterForward.ToString() + " | " + idealForward.ToString() + " | " + aimedLocation.ToString();

        #endregion


        /*
        return enemyLocation.GetType().ToString() + " | " + shooterLocation.GetType().ToString()

            + " >|< "

            + enemyLocation.ToString() + " | " + shooterLocation.ToString();
        */



        /*return agentPosition.GetType().ToString() + " | " + targetPosition.GetType().ToString()

            + " >|< "

            + agentPosition.ToString() + " | " + targetPosition.ToString();*/
    }
}
