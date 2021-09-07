using System.Collections;
using System.Collections.Generic;
using UnityEngine.VFX;
using UnityEngine;

[System.Serializable]
public abstract class TankControllerScript : MonoBehaviour
{
    private float turretRotation;
    private float gunPitch;
    private Transform cameraTransform;
    private float cameraPitch;
    protected GameObject hull;
    private GameObject turret;
    private GameObject gun;
    private Transform muzzle;
    public bool isPlayer = false;
    [SerializeField] private int hitPoints;
    private int fullHitpoints;
    public int tankIndex = 1;

    private VisualEffect muzzleSmokeVFX;
    private VisualEffect muzzleBlastVFX;
    private VisualEffect muzzleResidualSmokeVFX;
    private Light muzzleLightSource;

    private Camera aimCamera;
    private Camera thirdPersonCamera;
    private bool inAimView = false;
    private int cameraZoomFactor = 5;
    private GameManager gameManagerScript;

    protected bool reloaded = true;
    public TrainerScript AITrainer = null;
    public bool isDummy = false;

    public virtual void Start()
    {
        fullHitpoints = hitPoints;

        if (isDummy == false)
        {
            gameManagerScript = GameObject.FindGameObjectWithTag("GameController").GetComponent<GameManager>();
            hull = transform.Find("Objects/Hitbox/Hull").gameObject;
            turret = transform.Find("Objects/Hitbox/Turret").gameObject;
            gun = transform.Find("Objects/Hitbox/Turret/Gun").gameObject;
            muzzle = transform.Find("Objects/Hitbox/Turret/Gun/Muzzle");
            turretRotation = turret.transform.rotation.eulerAngles.y;
            gunPitch = gun.transform.rotation.eulerAngles.x;

            GameObject cameraTransformContainer = new GameObject();
            cameraTransformContainer.transform.position = gun.transform.position;
            cameraTransformContainer.transform.rotation = gun.transform.rotation;
            cameraTransformContainer.transform.parent = turret.transform;
            cameraTransformContainer.name = "<Camera Transform Container>";
            cameraTransform = cameraTransformContainer.transform;
            cameraPitch = cameraTransform.rotation.eulerAngles.x;

            muzzleSmokeVFX = transform.Find("Objects/Hitbox/Turret/Gun/Muzzle/Muzzle Smoke").GetComponent<VisualEffect>();
            muzzleSmokeVFX.enabled = false;
            muzzleBlastVFX = transform.Find("Objects/Hitbox/Turret/Gun/Muzzle/Muzzle Blast").GetComponent<VisualEffect>();
            muzzleBlastVFX.enabled = false;
            muzzleResidualSmokeVFX = transform.Find("Objects/Hitbox/Turret/Gun/Muzzle/Muzzle Residual Smoke").GetComponent<VisualEffect>();
            muzzleResidualSmokeVFX.enabled = false;
            muzzleLightSource = transform.Find("Objects/Hitbox/Turret/Gun/Muzzle/Muzzle Light Source").GetComponent<Light>();
            muzzleLightSource.enabled = false;

            if (isPlayer == true)
            {
                aimCamera = transform.Find("Objects/Hitbox/Turret/Gun/Aim Camera").GetComponent<Camera>();
                aimCamera.gameObject.SetActive(false);

                thirdPersonCamera = Camera.main;
            }
            else
            {

            }

            this.GetComponent<Rigidbody>().centerOfMass = new Vector3(0, -0.9f, 0);
        }
    }

    private void Update()
    {

    }

    /// <summary>
    /// This method handles the mouse input for all tanks.
    /// </summary>
    /// <param name="mouseSensitivity"></param>
    /// <param name="gunDepression"></param>
    /// <param name="gunElevation"></param>
    public void HandleMouseInput(float mouseSensitivity, float gunDepression, float gunElevation, float scrollZoomFactor)
    {
        turretRotation += mouseSensitivity * Input.GetAxis("Mouse X");
        gunPitch -= mouseSensitivity * Input.GetAxis("Mouse Y");
        cameraPitch -= (mouseSensitivity * Input.GetAxis("Mouse Y"));

        if (inAimView == false)
        {
            // Desync the gun aim and the camera aim just a little bit so the player can see what they are aiming at in third person viewwwwwwwwwwwww
            gunPitch = cameraPitch - 15;
        }

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

        if (inAimView == false)
        {
            cameraTransform.localRotation = Quaternion.Euler(cameraPitch, currentGunRotation.y, currentGunRotation.z);
            Camera.main.transform.rotation = cameraTransform.rotation;
            Camera.main.transform.position = turret.transform.position - (cameraTransform.forward * cameraZoomFactor * scrollZoomFactor);
        }

        cameraZoomFactor -= (int)Input.mouseScrollDelta.y;
        //Debug.Log(cameraZoomFactor);

        if (cameraZoomFactor < 2)
        {
            if (inAimView == false)
            {
                gameManagerScript.SwitchCamera(aimCamera);
                inAimView = true;
            }

            cameraZoomFactor = 1;
        }
        else if (cameraZoomFactor >= 2)
        {
            if (inAimView == true)
            {
                gameManagerScript.SwitchCamera(thirdPersonCamera);
                inAimView = false;
            }

            if (cameraZoomFactor > 6)
            {
                cameraZoomFactor = 6;
            }
        }

        //Debug.DrawRay(muzzle.position, muzzle.forward * 100, Color.red);
    }

    public void FireGun(GameObject round)
    {
        GameObject firedRound = Instantiate(round, muzzle.position, muzzle.rotation);
        firedRound.GetComponent<Rigidbody>().velocity = firedRound.transform.forward * (firedRound.GetComponent<ShellScript>().muzzleVelocity / 5f);

        PlayMuzzleVFX();
    }

    private void PlayMuzzleVFX()
    {
        muzzleBlastVFX.enabled = true;
        muzzleBlastVFX.Play();

        muzzleLightSource.enabled = true;
        StartCoroutine(StopMuzzleLight());

        muzzleSmokeVFX.enabled = true;
        muzzleSmokeVFX.SetVector3("Muzzle Direction", muzzle.forward * 2);
        muzzleSmokeVFX.SetVector3("Muzzle Up", muzzle.up);
        muzzleSmokeVFX.SetVector3("Muzzle Right", muzzle.right);
        muzzleSmokeVFX.Play();

        StartCoroutine(StopVFXGraph(muzzleSmokeVFX, 0.1f));
        StartCoroutine(StopVFXGraph(muzzleBlastVFX, 0.25f));

        muzzleResidualSmokeVFX.enabled = true;
        StartCoroutine(StartVFXGraph(muzzleResidualSmokeVFX, 1f));
        StartCoroutine(StopVFXGraph(muzzleResidualSmokeVFX, 4f));
    }

    private IEnumerator StopVFXGraph(VisualEffect vfx, float delayTime)
    {
        yield return new WaitForSeconds(delayTime);
        vfx.Stop();
    }

    private IEnumerator StartVFXGraph(VisualEffect vfx, float delayTime)
    {
        yield return new WaitForSeconds(delayTime);
        vfx.Play();
    }

    private IEnumerator StopMuzzleLight()
    {
        yield return new WaitForSeconds(0.5f);
        muzzleLightSource.enabled = false;
    }

    public List<string> SendArmourStateData(Vector3 globalArmourForward, Vector3 globalCollisionPoint, float penetrationDifference, Vector3 relativeAngle)
    {
        List<string> details = new List<string>();

        return details;
    }

    public int GetHitpoints()
    {
        return this.hitPoints;
    }

    public int GetMaxHitpoints()
    {
        return this.fullHitpoints;
    }

    // Only use this in training!
    public void ResetHitpoint()
    {
        hitPoints = fullHitpoints;
    }

    public void CauseDamage(int shellAlphaDamage)
    {
        if (hitPoints < shellAlphaDamage)
        {
            hitPoints = 0;
            DestroyTank();
        }
        else
        {
            hitPoints -= shellAlphaDamage;
        }
    }

    public void DestroyTank()
    {
        // TODO
    }

    public abstract void HandleAIMovement(int inputCommand);
}
