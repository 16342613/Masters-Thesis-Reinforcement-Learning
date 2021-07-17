using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class EnvironmentState : MonoBehaviour
{
    public Vector3 enemyLocation;
    public Vector3 enemyHullAngle;
    public Vector3 enemyTurretAngle;
    public int enemyHitpoints;

    public Vector3 shooterLocation;
    public float shooterPenetration;
    public string aimedPlate;
    public Vector3 aimedLocation;
    public float plateThickness;

    public EnvironmentState(Vector3 enemyLocation, Vector3 enemyHullAngle, Vector3 enemyTurretAngle, int enemyHitpoints, Vector3 shooterLocation, float shooterPenetration, string aimedPlate, Vector3 aimedLocation, float plateThickness)
    {
        this.enemyLocation = enemyLocation;
        this.enemyHullAngle = enemyHullAngle;
        this.enemyTurretAngle = enemyTurretAngle;
        this.enemyHitpoints = enemyHitpoints;
        this.shooterLocation = shooterLocation;
        this.shooterPenetration = shooterPenetration;
        this.aimedPlate = aimedPlate;
        this.aimedLocation = aimedLocation;
        this.plateThickness = plateThickness;
    }

    public void SetData(Vector3 enemyLocation, Vector3 enemyHullAngle, Vector3 enemyTurretAngle, int enemyHitpoints, Vector3 shooterLocation, float shooterPenetration, string aimedPlate, Vector3 aimedLocation, float plateThickness)
    {
        this.enemyLocation = enemyLocation;
        this.enemyHullAngle = enemyHullAngle;
        this.enemyTurretAngle = enemyTurretAngle;
        this.enemyHitpoints = enemyHitpoints;
        this.shooterLocation = shooterLocation;
        this.shooterPenetration = shooterPenetration;
        this.aimedPlate = aimedPlate;
        this.aimedLocation = aimedLocation;
        this.plateThickness = plateThickness;
    }


    public override string ToString()
    {
        return enemyLocation.GetType().ToString() + " | " + enemyHullAngle.GetType().ToString() + " | " + enemyTurretAngle.GetType().ToString() + " | " 
            + enemyHitpoints.GetType().ToString() + " | " + shooterLocation.GetType().ToString() + " | " + shooterPenetration.GetType().ToString() + " | " 
            + aimedLocation.GetType().ToString() + " | " + plateThickness.GetType().ToString()

            + " >|< " 
            
            + enemyLocation.ToString() + " | " + enemyHullAngle.ToString() + " | " + enemyTurretAngle.ToString() + " | " + enemyHitpoints.ToString() + " | "
            + shooterLocation.ToString() + " | " + shooterPenetration + " | " + aimedLocation.ToString() + " | " + plateThickness;
    }
}
