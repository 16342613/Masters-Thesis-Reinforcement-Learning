using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ShellScript : MonoBehaviour
{
    // Muzzle velocity in metres per second
    public float muzzleVelocity = 800;
    // Penetrative power in millimetres
    public float penetration = 175;
    private TankControllerScript originTank;

    public GameObject landing;

    // Start is called before the first frame update
    void Start()
    {

    }

    private void OnCollisionEnter(Collision collision)
    {
        Instantiate(landing, collision.GetContact(0).point, Quaternion.identity);
        Destroy(this.gameObject);


    }

    public TankControllerScript Get_originTank()
    {
        return originTank;
    }

    public void Set_originTank(TankControllerScript originTank)
    {
        this.originTank = originTank;
    }
}
