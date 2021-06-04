using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;

public class LightTank_ControllerScript : MonoBehaviour
{
    private List<WheelCollider> wheelColliders = new List<WheelCollider>();
    private List<GameObject> wheels = new List<GameObject>();
    public List<int> steeringWheelIndexes = new List<int>();
    public List<int> inverseSteeringWheelIndexes = new List<int>();
    public int maxSteeringAngle = 15;
    public int brakingPower = 100;
    public bool isPlayer = false;

    private GameObject turret;
    private GameObject gun;
    private float turretRotation;
    private float gunPitch;

    private GameManager gameManagerScript;


    // Start is called before the first frame update
    void Start()
    {
        wheelColliders = this.transform.Find("Wheel Colliders").GetComponentsInChildren<WheelCollider>().ToList();
        GameObject wheelsParent = this.transform.Find("Wheels").gameObject;
        foreach (Transform t in wheelsParent.transform)
        {
            wheels.Add(t.gameObject);
        }

        this.turret = this.transform.Find("Hitbox/Turret").gameObject;
        this.gun = this.transform.Find("Hitbox/Turret/Gun").gameObject;
        this.turretRotation = turret.transform.rotation.eulerAngles.y;
        this.gunPitch = gun.transform.rotation.eulerAngles.x;

        this.gameManagerScript = GameObject.FindGameObjectWithTag("GameController").GetComponent<GameManager>();
    }

    // Update is called once per frame
    void FixedUpdate()
    {
        if (this.isPlayer == true)
        {
            HandleMouseInput();
            HandleMovementInput();
        }
        UpdateWheelPositions();
    }

    private void UpdateWheelPositions()
    {
        for (int i = 0; i < wheels.Count; i++)
        {
            Vector3 wheelColliderPosition = Vector3.zero;
            Quaternion wheelColliderRotation = Quaternion.identity;

            wheelColliders[i].GetWorldPose(out wheelColliderPosition, out wheelColliderRotation);
            wheels[i].transform.position = wheelColliderPosition;
            wheels[i].transform.rotation = wheelColliderRotation;
        }
    }

    private void HandleMovementInput()
    {
        if (Input.GetKey(KeyCode.W))
        {
            for (int i = 0; i < wheelColliders.Count; i++)
            {
                wheelColliders[i].motorTorque = 30;
                wheelColliders[i].brakeTorque = 0;
            }
        }

        if (Input.GetKey(KeyCode.W) == false)
        {
            for (int i = 0; i < wheelColliders.Count; i++)
            {
                wheelColliders[i].motorTorque = 0;
                wheelColliders[i].brakeTorque = this.brakingPower;
            }
        }

        float steeringInput = this.maxSteeringAngle * Input.GetAxis("Horizontal");
        for (int i = 0; i < this.steeringWheelIndexes.Count; i++)
        {
            this.wheelColliders[this.steeringWheelIndexes[i]].steerAngle = steeringInput;
        }

        for (int i = 0; i < this.inverseSteeringWheelIndexes.Count; i++)
        {
            this.wheelColliders[this.inverseSteeringWheelIndexes[i]].steerAngle = -steeringInput;
        }
    }

    private void HandleMouseInput()
    {
        this.turretRotation += this.gameManagerScript.mouseSensitivity * Input.GetAxis("Mouse X");
        this.gunPitch -= this.gameManagerScript.mouseSensitivity * Input.GetAxis("Mouse Y") * 1f;

        Vector3 currentTurretRotation = this.turret.transform.rotation.eulerAngles;
        this.turret.transform.rotation = Quaternion.Euler(currentTurretRotation.x, turretRotation, currentTurretRotation.z);

        Vector3 currentGunRotation = this.gun.transform.rotation.eulerAngles;
        this.gun.transform.rotation = Quaternion.Euler(gunPitch, currentGunRotation.y, currentGunRotation.z);
    }
}
