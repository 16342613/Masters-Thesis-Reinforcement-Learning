using System.Collections;
using System.Collections.Generic;
using UnityEngine;

enum ServerRequests
{
    // Given an observation, the server returns an action which follows the policy
    PREDICT,
    // Build up the replay buffer and learn the optimal policy
    BUILD_BUFFER,
    // Train the prediction network
    TRAIN,
    // Set the discount factor (gamma)
    SET_DISCOUNT_FACTOR,
    // Update the target neural network on the server
    UPDATE_TARGET_NETWORK,
    // Test connection
    TEST_CONNECTION,
    // Echo a string on the server's console
    ECHO,
    // Plot the rewards and steps on the server
    PLOT,
    // Save the neural networks on the server
    SAVE_NETWORKS,
    // Updates the reward for the state in the replay buffer when given the state ID
    UPDATE_REWARD,

    // Send transition data to the A3C server
    SEND_A3C_TRANSITION,
    // Send the transition data to the A3C server and get the action probability distribution
    SEND_TRANSITION_GET_ACTION
}

public abstract class TrainerScript : MonoBehaviour
{
    public string AIName = "Sole Trainer";

    public CommunicationClient client;
    public float epsilon = 1f;
    public List<float> episodeRewards = new List<float>();

    public MasterTrainerScript masterTrainer;

    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    public abstract object[] TakeAction(EnvironmentState state, int actionRepeat = 1, string givenResponse = null);

    public abstract void ResetEnvironment();

    public void TestConnection()
    {
        string response = client.RequestResponse(ServerRequests.TEST_CONNECTION.ToString() + " >|< Connection test! ");
        Debug.Log(response);
    }
}
