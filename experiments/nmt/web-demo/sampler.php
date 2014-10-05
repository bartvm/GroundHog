<?php
/* This basically forwards API requests to Flower. Needed if
 * the webserver doesn't support Python, but does have PHP support. */

$api = 'http://localhost:5555/api/';

$ch = curl_init();
curl_setopt($ch, CURLOPT_RETURNTRANSFER, 1);

if ($_SERVER["REQUEST_METHOD"] == "POST") {
  curl_setopt($ch, CURLOPT_HTTPHEADER, array('Content-Type: application/json'));
  curl_setopt($ch, CURLOPT_POST, 1);
  curl_setopt($ch, CURLOPT_URL, $api . $_POST['method']);
  curl_setopt($ch, CURLOPT_POSTFIELDS, $_POST['json']);
} elseif ($_SERVER["REQUEST_METHOD"] == "GET") {
  curl_setopt($ch, CURLOPT_URL, $api . $_GET['method']);
} else {
  echo '{}';
  exit;
}
echo curl_exec($ch);
curl_close($ch);
?>

