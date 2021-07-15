using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class EnvironmentState : MonoBehaviour
{
    public Vector3 enemyLocation;
    public Vector3 enemyHullAngle;
    public Vector3 enemyTurretAngle;
    // public float hitpints;

    public Vector3 shooterLocation;
    public float shooterPenetration;
    public string aimedPlate;
    public Vector3 aimedLocation;
    public float plateThickness;

    public EnvironmentState(Vector3 enemyLocation, Vector3 enemyHullAngle, Vector3 enemyTurretAngle, Vector3 shooterLocation, float shooterPenetration, string aimedPlate, Vector3 aimedLocation, float plateThickness)
    {
        this.enemyLocation = enemyLocation;
        this.enemyHullAngle = enemyHullAngle;
        this.enemyTurretAngle = enemyTurretAngle;
        this.shooterLocation = shooterLocation;
        this.shooterPenetration = shooterPenetration;
        this.aimedPlate = aimedPlate;
        this.aimedLocation = aimedLocation;
        this.plateThickness = plateThickness;
    }

    public void SetData(Vector3 enemyLocation, Vector3 enemyHullAngle, Vector3 enemyTurretAngle, Vector3 shooterLocation, float shooterPenetration, string aimedPlate, Vector3 aimedLocation, float plateThickness)
    {
        this.enemyLocation = enemyLocation;
        this.enemyHullAngle = enemyHullAngle;
        this.enemyTurretAngle = enemyTurretAngle;
        this.shooterLocation = shooterLocation;
        this.shooterPenetration = shooterPenetration;
        this.aimedPlate = aimedPlate;
        this.aimedLocation = aimedLocation;
        this.plateThickness = plateThickness;
    }


    public override string ToString()
    {
        return enemyLocation.ToString() + " | " + enemyHullAngle.ToString() + " | " + enemyTurretAngle.ToString() + " | " + 
            shooterLocation.ToString() + " | " + shooterPenetration + " | " + aimedPlate + " | " + aimedLocation.ToString() + " | " + plateThickness;
    }
}
