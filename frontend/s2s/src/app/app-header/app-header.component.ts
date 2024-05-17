import { Component } from '@angular/core';
import { Input } from '@angular/core';

/**
 * Defines the header component that is used to display the title and subtitle
 * of the application as well as holds the navigation bar.
 *
 * @summary Header component that displays the title and subtitle of the application.
 *
 * @example
 * <app-header title="Title" subtitle="Subtitle"></app-header>
 */
@Component({
  selector: 'app-header',
  templateUrl: './app-header.component.html',
  styleUrl: './app-header.component.css'
})
export class AppHeaderComponent {
  // accepts as input the title and subtitle of the given page
  @Input() title?: string;
  @Input() subtitle?: string;

  constructor() { }
}
