import fire, time
from openai import OpenAI


def main(file_path, file_path_out, batch_id=None):
    client = OpenAI()

    if batch_id is None:
        batch_input_file = client.files.create(
            file=open(file_path, "rb"), purpose="batch"
        )

        batch_input_file_id = batch_input_file.id

        batch = client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "description": "nightly eval job"
            }
        )
        batch_id = batch.id
    print(f"working with batch (id={batch_id})")

    while True:
       batch = client.batches.retrieve(batch_id)
       if batch.status in ["failed", "completed", "expired", "cancelled"]:
          break
       print("waiting for batch to finish...")
       time.sleep(10)

    if batch.status == "completed":
       print("batch completed!!!")
       content = client.files.content(batch.output_file_id)
       with open(file_path_out, "w") as fout:
            fout.write(content)


if __name__ == '__main__':
  fire.Fire(main)