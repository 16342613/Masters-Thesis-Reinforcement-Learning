using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public abstract class TankControllerScript : MonoBehaviour
{
    private float turretRotation;
    private float gunPitch;
    private Transform cameraTransform;
    private float cameraPitch;
    private GameObject turret;
    private GameObject gun;
    private Transform muzzle;
    public bool isPlayer = false;

    public virtual void Start()
    {
        turret = transform.Find("Hitbox/Turret").gameObject;
        gun = transform.Find("Hitbox/Turret/Gun").gameObject;
        muzzle = transform.Find("Hitbox/Turret/Gun/Muzzle");
        turretRotation = turret.transform.rotation.eulerAngles.y;
        gunPitch = gun.transform.rotation.eulerAngles.x;

        GameObject cameraTransformContainer = new GameObject();
        cameraTransformContainer.transform.position = gun.transform.position;
        cameraTransformContainer.transform.rotation = gun.transform.rotation;
        cameraTransformContainer.transform.parent = turret.transform;
        cameraTransformContainer.name = "<Camera Transform Container>";
        cameraTransform = cameraTransformContainer.transform;
        cameraPitch = cameraTransform.rotation.eulerAngles.x;
    }

    /// <summary>
    /// This method handles the mouse input for all tanks.
    /// </summary>
    /// <param name="mouseSensitivity"></param>
    /// <param name="gunDepression"></param>
    /// <param name="gunElevation"></param>
    public void HandleMouseInput(float mouseSensitivity, float gunDepression, float gunElevation)
    {
        turretRotation += mouseSensitivity * Input.GetAxis("Mouse X");
        gunPitch -= mouseSensitivity * Input.GetAxis("Mouse Y");
        cameraPitch -= mouseSensitivity * Input.GetAxis("Mouse Y");

        gunPitch = cameraPitch;
        // Clamp the gun between the maximum and minimum gun depression angles
        if (gunPitch > -gunDepression)
        {
            gunPitch = -gunDepression;
        }
        else if (gunPitch < -gunElevation)
        {
            gunPitch = -gunElevation;
        }

        // The camera should have more longitudinal freedom, but still clamp them between 90 and 0 degrees
        if (cameraPitch < -90)
        {
            cameraPitch = -90;
        }
        else if (cameraPitch > 90)
        {
            cameraPitch = 90;
        }

        Vector3 currentTurretRotation = turret.transform.localEulerAngles;
        turret.transform.localRotation = Quaternion.Euler(currentTurretRotation.x, turretRotation, currentTurretRotation.z);

        Vector3 currentGunRotation = gun.transform.localEulerAngles;
        gun.transform.localRotation = Quaternion.Euler(gunPitch, currentGunRotation.y, currentGunRotation.z);

        cameraTransform.localRotation = Quaternion.Euler(cameraPitch, currentGunRotation.y, currentGunRotation.z);
        Camera.main.transform.rotation = cameraTransform.rotation;
        Camera.main.transform.position = turret.transform.position - (cameraTransform.forward * 10);

        Debug.Log(gunPitch);
    }

    public void FireGun(GameObject round)
    {
        Debug.DrawRay(gun.transform.position, gun.transform.forward * 100);

        GameObject firedRound = Instantiate(round, muzzle.position, muzzle.rotation);
        firedRound.GetComponent<Rigidbody>().velocity = firedRound.transform.forward * (firedRound.GetComponent<ShellScript>().muzzleVelocity / 5f);
    }
}
