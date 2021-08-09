using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;

using CielaSpike;

public class MasterTrainerScript : MonoBehaviour
{
    List<GameObject> environments = new List<GameObject>();
    // Change the trainer script type between <> accordingly. In the end, you only need one trainer
    List<ArmourTrainerAI> trainingScripts = new List<ArmourTrainerAI>();

    public int environmentCount;
    public GameObject environment;
    public Vector3 offset = new Vector3(0, 0, 10);

    // Start is called before the first frame update
    void Start()
    {
        SetUpEnvironments();

        environments = GameObject.FindGameObjectsWithTag("Environment").ToList();
        foreach (GameObject environment in environments)
        {
            // This may be null depending on the type of trainer. In the end, you should only use one trainer
            trainingScripts.Add(environment.GetComponentInChildren<ArmourTrainerAI>());
        }

        for (int i = 0; i < trainingScripts.Count; i++)
        {
            trainingScripts[i].masterTrainer = this;
            trainingScripts[i].AIName = i.ToString();
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

        // Debug.Log(GlobalScript.globalEpisodeCount);
    }

    private void SetUpEnvironments()
    {
        Vector3 currentSpawnPoint = Vector3.zero;

        for (int i = 0; i < environmentCount; i++)
        {
            GameObject newEnvironment = GameObject.Instantiate(environment);
            newEnvironment.transform.position = currentSpawnPoint;

            currentSpawnPoint += offset;
        }
    }

    private void Train()
    {
        foreach (ArmourTrainerAI trainingScript in trainingScripts)
        {
            StartCoroutine(trainingScript.TrainA3C());
        }
    }

    private void TrainAsyncronous()
    {

    }
}
