export function onError(error) {
    let message = error.toString();
    let errorCode = message.split("status code ")[1]

    // API Gateway timeout errors
    if (errorCode == "504" || errorCode == "500") {
      let coldStart = true;
      return coldStart;
    }

    // Auth errors
    else if (!(error instanceof Error) && error.message) {
      message = error.message;
    }
    alert(message);
    return false;
  }