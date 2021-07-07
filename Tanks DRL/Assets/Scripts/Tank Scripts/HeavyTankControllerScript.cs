using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class HeavyTankControllerScript : TankControllerScript
{
    public float gunDepression = -8;
    public float gunElevation = 25;
    public List<GameObject> shellTypes = new List<GameObject>();
    public float scrollZoomFactor = 2.5f;
    public float horsepower = 50;
    public float hullTraverseSpeed = 40;
    public float topSpeed = 35;
    public float reverseSpeed = 15;

    private Rigidbody tankRigidBody;

    public HeavyTankControllerScript() : base()
    {
        // Empty Constructor
    }

    // Start is called before the first frame update
    public override void Start()
    {
        base.Start();   // Call the Start method of the superclass

        tankRigidBody = this.GetComponent<Rigidbody>();
    }

    // Update is called once per frame
    void FixedUpdate()
    {
        if (tankRigidBody.velocity.magnitude > topSpeed)
        {
            tankRigidBody.velocity = tankRigidBody.velocity.normalized * topSpeed;
        }

        Debug.Log(tankRigidBody.velocity.magnitude);

        if (isPlayer == true)
        {
            base.HandleMouseInput(2f, gunDepression, gunElevation, scrollZoomFactor);
            HandleMovementInput();

            if (Input.GetKey(KeyCode.X))
            {
                base.FireGun(shellTypes[0]);
            }
        }
    }

    private void HandleMovementInput()
    {
        Vector3 currentHullRotation = hull.transform.localEulerAngles;

        if (Input.GetKey(KeyCode.W))
        {
            tankRigidBody.AddForce(hull.transform.TransformDirection(new Vector3(0, 0, horsepower)));
        }
        else if (Input.GetKey(KeyCode.S))
        {
            tankRigidBody.AddForce(hull.transform.TransformDirection(new Vector3(0, 0, -horsepower)));
        }

        if (Input.GetKey(KeyCode.A))
        {
            hull.transform.localEulerAngles = new Vector3(currentHullRotation.x, currentHullRotation.y - (hullTraverseSpeed * Time.deltaTime), currentHullRotation.z);
        }
        else if (Input.GetKey(KeyCode.D))
        {
            hull.transform.localEulerAngles = new Vector3(currentHullRotation.x, currentHullRotation.y + (hullTraverseSpeed * Time.deltaTime), currentHullRotation.z);
        }
    }
}
