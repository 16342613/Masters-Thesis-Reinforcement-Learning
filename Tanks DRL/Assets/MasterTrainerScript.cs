using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;

public class MasterTrainerScript : MonoBehaviour
{
    List<GameObject> environments = new List<GameObject>();
    List<ArmourTrainerAI> trainingScripts = new List<ArmourTrainerAI>();

    public int environmentCount;
    public GameObject environment;

    // Start is called before the first frame update
    void Start()
    {
        SetUpEnvironments();

        environments = GameObject.FindGameObjectsWithTag("Environment").ToList();
        foreach (GameObject environment in environments)
        {
            trainingScripts.Add(environment.GetComponentInChildren<ArmourTrainerAI>());
        }
    }

    // Update is called once per frame
    void Update()
    {
        if (Input.GetKeyDown(KeyCode.M))
        {
            foreach (ArmourTrainerAI trainingScript in trainingScripts)
            {
                trainingScript.TestConnection();
            }
        }

        if (Input.GetKeyDown(KeyCode.B))
        {
            Train();
        }
    }

    private void SetUpEnvironments()
    {
        Vector3 currentSpawnPoint = Vector3.zero;

        for (int i = 0; i < environmentCount; i++)
        {
            GameObject newEnvironment = GameObject.Instantiate(environment);
            newEnvironment.transform.position = currentSpawnPoint;

            currentSpawnPoint += new Vector3(0, 0, 10);
        }
    }

    private void Train()
    {
        foreach(ArmourTrainerAI trainingScript in trainingScripts)
        {
            StartCoroutine(trainingScript.Train());
        }
    }
}
